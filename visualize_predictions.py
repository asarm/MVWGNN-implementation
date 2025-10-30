"""
Visualization Script for Wind Speed Predictions
Generates predictions for the last 50 time steps for a selected city.
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import DDGNNWind
from dataLoad import load_data
from torchData import GraphWeatherDataset, create_dataloader
import os


def load_trained_model(checkpoint_path, n_stations, hidden_dim=64, n_heads=4, seq_len=24, n_gnn_layers=2):
    """Load the trained model from checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = DDGNNWind(
        n_stations=n_stations,
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        seq_len=seq_len,
        n_gnn_layers=n_gnn_layers
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded from {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Best validation loss: {checkpoint['best_val_loss']:.6f}")
    
    return model, device


def generate_predictions(model, data_loader, device, n_steps=50):
    """Generate predictions for the last n_steps."""
    model.eval()
    
    # Multi-horizon prediction horizons
    prediction_horizons = [1, 3, 6, 12, 24]
    
    # Get scaler from the dataset for inverse transform
    scaler = data_loader.dataset.wind_speed_scaler
    scaler_mean = torch.from_numpy(scaler.mean_).float().to(device)  # (n_stations,)
    scaler_scale = torch.from_numpy(scaler.scale_).float().to(device)  # (n_stations,)
    
    all_predictions = {h: [] for h in prediction_horizons}
    all_targets = {h: [] for h in prediction_horizons}
    
    # Collect predictions for all batches
    for batch_idx, data in enumerate(data_loader):
        (
            historical_data,
            lat,
            lon,
            current_hour,
            day_of_year,
            wind_dir_deg_last,
            positions,
            target,
        ) = data

        historical_data = historical_data.to(device)
        lat = lat.to(device)
        lon = lon.to(device)
        current_hour = current_hour.to(device)
        day_of_year = day_of_year.to(device)
        wind_dir_deg_last = wind_dir_deg_last.to(device)
        positions = positions.to(device)
        target = target.to(device)

        with torch.no_grad():
            output = model(
                historical_data,
                lat,
                lon,
                current_hour,
                day_of_year,
                wind_dir_deg_last,
                positions,
            )
            
            # output: (B, N, 5), target: (B, N, 5)
            # Inverse transform predictions and targets back to original scale
            output_original = output * scaler_scale.unsqueeze(0).unsqueeze(-1) + scaler_mean.unsqueeze(0).unsqueeze(-1)
            target_original = target * scaler_scale.unsqueeze(0).unsqueeze(-1) + scaler_mean.unsqueeze(0).unsqueeze(-1)

        # Store predictions and targets for each horizon
        for i, h in enumerate(prediction_horizons):
            all_predictions[h].append(output_original[:, :, i].cpu().numpy())  # (B, N)
            all_targets[h].append(target_original[:, :, i].cpu().numpy())  # (B, N)
    
    # Concatenate all batches for each horizon
    for h in prediction_horizons:
        all_predictions[h] = np.concatenate(all_predictions[h], axis=0)  # (total_samples, n_stations)
        all_targets[h] = np.concatenate(all_targets[h], axis=0)  # (total_samples, n_stations)
    
        # Get last n_steps
        all_predictions[h] = all_predictions[h][-n_steps:]
        all_targets[h] = all_targets[h][-n_steps:]
    
    return all_predictions, all_targets


def plot_city_predictions(predictions_dict, targets_dict, city_name, city_idx, n_steps, save_path=None):
    """Plot predictions vs targets for a specific city across all horizons."""
    
    prediction_horizons = [1, 3, 6, 12, 24]
    
    # Create subplots for each horizon
    fig, axes = plt.subplots(len(prediction_horizons), 1, figsize=(14, 4*len(prediction_horizons)))
    if len(prediction_horizons) == 1:
        axes = [axes]
    
    metrics = {}
    
    for idx, h in enumerate(prediction_horizons):
        # Extract data for the selected city and horizon
        city_predictions = predictions_dict[h][:, city_idx]
        city_targets = targets_dict[h][:, city_idx]
        
        # Calculate metrics
        mae = np.mean(np.abs(city_predictions - city_targets))
        rmse = np.sqrt(np.mean((city_predictions - city_targets) ** 2))
        metrics[h] = {'mae': mae, 'rmse': rmse}
        
        # Create plot for this horizon
        time_steps = range(len(city_predictions))
        
        axes[idx].plot(time_steps, city_targets, 'b-', label='Gerçek Değer (Target)', 
                      linewidth=2, alpha=0.8, marker='o', markersize=4)
        axes[idx].plot(time_steps, city_predictions, 'r--', label='Tahmin (Prediction)', 
                      linewidth=2, alpha=0.8, marker='s', markersize=4)
        
        axes[idx].set_xlabel('Zaman Adımı', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Rüzgar Hızı (m/s)', fontsize=12, fontweight='bold')
        axes[idx].set_title(f'{h}-Saat Tahmin - MAE: {mae:.4f} m/s | RMSE: {rmse:.4f} m/s', 
                           fontsize=13, fontweight='bold')
        axes[idx].legend(fontsize=10, loc='best')
        axes[idx].grid(True, alpha=0.3, linestyle='--')
    
    plt.suptitle(f'{city_name} - Son {n_steps} Adım İçin Multi-Horizon Rüzgar Hızı Tahmini', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Grafik kaydedildi: {save_path}")
    
    plt.show()
    
    return metrics


def main():
    # Configuration
    checkpoint_path = "checkpoints/best_model.pth"
    seq_len = 24
    hidden_dim = 64  # Updated to match training configuration
    prediction_horizons = [1, 3, 6, 12, 24]  # Multi-horizon
    n_steps = 50  # Son kaç adımı görselleştireceğiz
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"❌ Hata: Model dosyası bulunamadı: {checkpoint_path}")
        print("Lütfen önce modeli eğitin (main.py)")
        return
    
    # Load data
    print("Veri yükleniyor...")
    df, cities = load_data("hourly-data")
    print(f"✓ Veri yüklendi: {len(df)} zaman adımı, {len(cities)} şehir")
    
    # Print available cities
    print("\n" + "="*60)
    print("Mevcut Şehirler:")
    print("="*60)
    for idx, city_name in enumerate(cities['City'].tolist()):
        print(f"{idx:2d}. {city_name}")
    print("="*60)
    
    # Get user input for city selection
    while True:
        try:
            city_idx = int(input(f"\nLütfen şehir numarasını seçin (0-{len(cities)-1}): "))
            if 0 <= city_idx < len(cities):
                break
            else:
                print(f"❌ Hata: Lütfen 0 ile {len(cities)-1} arasında bir sayı girin.")
        except ValueError:
            print("❌ Hata: Lütfen geçerli bir sayı girin.")
    
    city_name = cities['City'].iloc[city_idx]
    print(f"\n✓ Seçilen şehir: {city_name} (İndeks: {city_idx})")
    
    # Setup dataset and dataloader
    n_stations = len(cities)
    train_size = int(0.8 * len(df))
    
    print(f"\nTest verisi hazırlanıyor...")
    test_dataset = GraphWeatherDataset(
        df=df.iloc[train_size:],
        cities=cities,
        seq_len=seq_len,
        prediction_window=max(prediction_horizons),  # Max horizon
        sliding_window=True,
    )
    test_loader = create_dataloader(test_dataset, batch_size=1, shuffle=False)
    print(f"✓ Test verisi hazır: {len(test_dataset)} örnek")
    
    # Load model
    print("\nModel yükleniyor...")
    model, device = load_trained_model(
        checkpoint_path=checkpoint_path,
        n_stations=n_stations,
        hidden_dim=hidden_dim,
        seq_len=seq_len
    )
    
    # Generate predictions
    print(f"\nSon {n_steps} adım için tahminler üretiliyor...")
    predictions_dict, targets_dict = generate_predictions(model, test_loader, device, n_steps=n_steps)
    print(f"✓ Tahminler üretildi - Her horizon için: {predictions_dict[1].shape}")
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Plot predictions
    print(f"\n{city_name} için tahminler görselleştiriliyor...")
    save_path = f"results/prediction_{city_name.replace(' ', '_')}_last_{n_steps}_steps_multihorizon.png"
    metrics = plot_city_predictions(
        predictions_dict=predictions_dict,
        targets_dict=targets_dict,
        city_name=city_name,
        city_idx=city_idx,
        n_steps=n_steps,
        save_path=save_path
    )
    
    # Print summary statistics
    print("\n" + "="*60)
    print(f"Özet İstatistikler - {city_name}")
    print("="*60)
    print(f"Zaman Adımları: {n_steps}")
    for h in prediction_horizons:
        print(f"\n{h}-Saat Tahmin:")
        print(f"  MAE (Ortalama Mutlak Hata): {metrics[h]['mae']:.4f} m/s")
        print(f"  RMSE (Kök Ortalama Kare Hata): {metrics[h]['rmse']:.4f} m/s")
        print(f"  Ortalama Gerçek Değer: {targets_dict[h][:, city_idx].mean():.4f} m/s")
        print(f"  Ortalama Tahmin: {predictions_dict[h][:, city_idx].mean():.4f} m/s")
    print("="*60)
    
    # ========== MODEL SAĞLAMLIK TESTİ ==========
    # Modelin gerçekten öğrenip öğrenmediğini test et
    # Her horizon için ayrı ayrı test yap
    print("\n" + "="*60)
    print("MODEL SAĞLAMLIK TESTİ - Multi-Horizon")
    print("="*60)
    print("Her horizon için modelin performansını kontrol ediyoruz...")
    print()
    
    for h in prediction_horizons:
        # Seçilen şehir için verileri al
        city_predictions = predictions_dict[h][:, city_idx]
        city_targets = targets_dict[h][:, city_idx]
        
        # Orijinal MAE (model tahminleri)
        original_mae = np.mean(np.abs(city_predictions - city_targets))
        
        # Tahminleri bir geri kaydır (ilk tahmin atılır, sonuna NaN eklenir)
        shifted_predictions = np.roll(city_predictions, -1)
        
        # Son elemanı çıkar çünkü kaydırma sonrası eşleşmiyor
        shifted_predictions = shifted_predictions[:-1]
        shifted_targets = city_targets[:-1]
        
        # Kaydırılmış MAE hesapla
        shifted_mae = np.mean(np.abs(shifted_predictions - shifted_targets))
        
        print(f"\n📊 {h}-Saat Tahmin:")
        print(f"   Orijinal Model MAE:     {original_mae:.6f} m/s")
        print(f"   Kaydırılmış Tahmin MAE: {shifted_mae:.6f} m/s")
        
        # Sonuç yorumlama
        if original_mae < shifted_mae:
            improvement = ((shifted_mae - original_mae) / shifted_mae) * 100
            print(f"   ✅ BAŞARILI! Model {improvement:.2f}% daha iyi.")
        else:
            worsening = ((original_mae - shifted_mae) / original_mae) * 100
            print(f"   ⚠️  Model önceki değeri kopyalıyor olabilir ({worsening:.2f}% daha kötü).")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()

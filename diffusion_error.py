import torch
import torch.nn as nn
import numpy as np

# Generate clean signals (BPSK symbols)
def generate_signals(num_samples, seq_length):
    symbols = np.random.choice([-1, 1], size=(num_samples, seq_length))
    return torch.tensor(symbols, dtype=torch.float32)

# Diffusion process (forward corruption)
def forward_diffusion(x, timestep, total_timesteps=100):
    alpha = 1.0 - timestep / total_timesteps
    noise = torch.randn_like(x) * np.sqrt(1 - alpha)
    return np.sqrt(alpha) * x + noise, noise

# Denoising model (U-Net for 1D signals)
class Denoiser(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(Denoiser, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mid = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x, t):
        # Add timestep embedding
        t_embed = torch.ones_like(x) * (t / 100)
        x = torch.cat([x, t_embed], dim=1)
        
        # Encoder
        enc = self.encoder(x)
        mid = self.mid(enc)
        dec = self.decoder(mid)
        return dec

# Diffusion training
def train_diffusion(model, clean_data, epochs=500, batch_size=32, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    seq_length = clean_data.shape[1]
    total_timesteps = 100
    
    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(0, len(clean_data), batch_size):
            # Sample batch
            batch = clean_data[i:i+batch_size]
            
            # Random timestep
            t = torch.randint(1, total_timesteps, (1,)).item()
            
            # Forward diffusion
            corrupted, noise = forward_diffusion(batch, t, total_timesteps)
            
            # Predict noise
            predicted_noise = model(corrupted, t)
            
            # Compute loss
            loss = criterion(predicted_noise, noise)
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if (epoch+1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(clean_data):.4f}')
    
    return model

# Reverse diffusion for error correction
def reverse_diffusion(model, corrupted_signal, total_timesteps=100):
    x = corrupted_signal
    for t in range(total_timesteps-1, 0, -1):
        # Predict noise
        with torch.no_grad():
            noise_pred = model(x, t)
        
        # Reverse step
        alpha = 1.0 - t / total_timesteps
        x = (x - np.sqrt(1 - alpha) * noise_pred) / np.sqrt(alpha)
        x = torch.clamp(x, -1, 1)  # Clip to valid BPSK range
    
    return x

# Main execution
if __name__ == "__main__":
    # Parameters
    num_samples = 1000
    seq_length = 64  # Signal length
    snr_db = 10  # Signal-to-noise ratio
    
    # Generate clean BPSK signals
    clean_signals = generate_signals(num_samples, seq_length)
    
    # Create model
    model = Denoiser(seq_length + 1)  # +1 for timestep embedding
    
    # Train model
    model = train_diffusion(model, clean_signals)
    
    # Test with a noisy signal
    test_signal = clean_signals[0].unsqueeze(0)
    corrupted, _ = forward_diffusion(test_signal, 90)  # Highly corrupted
    
    # Recover signal
    recovered = reverse_diffusion(model, corrupted)
    
    print("Original signal:", test_signal[0, :10])
    print("Corrupted signal:", corrupted[0, :10])
    print("Recovered signal:", recovered[0, :10])

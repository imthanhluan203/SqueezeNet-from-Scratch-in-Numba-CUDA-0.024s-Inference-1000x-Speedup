"""
Main execution script for SqueezeNet CUDA training
"""
from .utils import setup_seed, print_versions
from .data import load_datasets, create_data_loaders, print_dataset_info
from .train import train_model

def main():
    """Main function to run the training"""
    # Setup
    setup_seed()
    print_versions()
    
    # Load data
    print("Loading datasets...")
    train_ds, test_ds, categories = load_datasets()
    train_loader, val_loader = create_data_loaders(train_ds, test_ds)
    print_dataset_info(train_loader, val_loader, categories)
    
    # Train model
    print("Starting training...")
    weights_np, loss_history, acc_history = train_model(train_loader)
    
    print("Training completed!")
    print(f"Final loss: {loss_history[-1]:.4f}")
    print(f"Final accuracy: {acc_history[-1]:.4f}")
    
    return weights_np, loss_history, acc_history

if __name__ == "__main__":
    main()

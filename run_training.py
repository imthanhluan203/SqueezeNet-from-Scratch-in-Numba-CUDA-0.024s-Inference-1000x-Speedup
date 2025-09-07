#!/usr/bin/env python3
"""
Script để chạy training SqueezeNet CUDA từ các file đã được chia nhỏ
"""
import sys
import os

# Thêm thư mục src vào Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Main function"""
    try:
        from src.main import main as train_main
        print("Starting SqueezeNet CUDA Training...")
        weights, loss_history, acc_history = train_main()
        print("Training completed successfully!")
        return weights, loss_history, acc_history
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please make sure all dependencies are installed:")
        print("pip install -r requirements.txt")
        return None, None, None
    except Exception as e:
        print(f"Error during training: {e}")
        return None, None, None

if __name__ == "__main__":
    main()

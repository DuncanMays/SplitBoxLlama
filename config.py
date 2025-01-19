# Update the MASTER_CONFIG with batch_size and context_window parameters
MASTER_CONFIG = {
    'batch_size': 32,        # Number of batches to be processed at each random split
    'context_window': 16,    # Number of characters in each input (x) and target (y) sequence of each batch
    'd_model': 128,          # dimension of linear layers
    'epochs': 5000//2,          # Number of training epochs
    'log_interval': 500,      # Log information every 10 batches during training
    'n_layers': 4,
    'n_heads': 8,
}
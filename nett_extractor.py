import glob
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import os


TAU_VALUES = {
    1: 0.0, 2: 1.0, 3: 5.0, 4: 10.0, 5: 20.0, 
    6: 40.0, 7: 60.0, 8: 80.0, 9: 100.0
}

PREV_ADVISORY_VALUES = {
    1: 0.0, 2: 1.0, 3: 2.0, 4: 3.0, 5: 4.0
}





def generate_unified_dataset(nnet_folder, samples_per_file=10000):
    """
    Genera un dataset unificato iterando su tutti i file .nnet nella cartella.
    Aggiunge due colonne all'input: Tau e PrevAdvisory.
    """
    all_X = []
    all_Y = []
    
    # Cerca pattern ACASXU*.nnet
    search_path = os.path.join(nnet_folder, "*.nnet")
    files = glob.glob(search_path)
    
    if not files:
        raise ValueError(f"Nessun file .nnet trovato in {nnet_folder}")
        
    print(f"--- Inizio generazione Super-Dataset da {len(files)} reti ---")
    
    for filepath in sorted(files):
        filename = os.path.basename(filepath)
        
        # Parsing nome file: ACASXU_run2a_X_Y_batch_2000.nnet
        parts = filename.split('_')
        
        try:
            # Assumiamo formato standard VNN-COMP/Reluplex
            # parts[2] è X (Prev Advisory Index)
            # parts[3] è Y (Tau Index)
            idx_prev = int(parts[2])
            idx_tau = int(parts[3])
        except (ValueError, IndexError):
            print(f"SKIPPING: {filename} non rispetta il formato ACASXU_run2a_X_Y_...")
            continue

        # Recupera valori reali dalle tabelle di lookup
        if idx_tau not in TAU_VALUES or idx_prev not in PREV_ADVISORY_VALUES:
            print(f"SKIPPING: {filename} ha indici fuori range (Prev={idx_prev}, Tau={idx_tau})")
            continue
            
        tau_val = TAU_VALUES[idx_tau]
        prev_val = PREV_ADVISORY_VALUES[idx_prev]
        
        print(f"Processando {filename} -> [Prev: {prev_val}, Tau: {tau_val}]")

        # 1. Genera dati 5D usando l'oracolo specifico
        oracle = ACASXuOracle(filepath)
        X_batch, Y_batch = oracle.generate_dataset(samples_per_file)
        
        # 2. Aggiungi colonne Tau e Prev
        N = X_batch.shape[0]
        # Creiamo vettori colonna
        tau_col = np.full((N, 1), tau_val)
        prev_col = np.full((N, 1), prev_val)
        
        # Stack orizzontale: Input diventa 7D
        # Ordine: [rho, theta, psi, v_own, v_int, tau, prev]
        X_augmented = np.hstack((X_batch, tau_col, prev_col))
        
        all_X.append(X_augmented)
        all_Y.append(Y_batch)

    if not all_X:
        raise ValueError("Nessun dato generato.")

    final_X = np.vstack(all_X)
    final_Y = np.vstack(all_Y)
    
    return final_X, final_Y



class ACASXuOracle:
    def __init__(self, filename):
        self.filename = filename
        self.weights = []
        self.biases = []
        self.means = []
        self.ranges = []
        self.min_inputs = []
        self.max_inputs = []
        self._load_nnet()

    def _to_float_vector(self, seq, name: str):
        """Convert to 1D float vector, raise if not numeric."""
        cleaned = [x for x in seq if str(x).strip() != ""]
        arr = np.asarray(cleaned, dtype=float)
        if arr.ndim != 1:
            raise ValueError(f"{name} deve essere un vettore 1D")
        return arr

    def _load_nnet(self):
        """Parse the custom .nnet format."""
        with open(self.filename, 'r') as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith("//")]

        # Header parsing
        # Line 0: num_layers, input_size, output_size, max_layer_size
        try:
            header = [int(x) for x in lines[0].split(",") if x.strip()]
        except ValueError as exc:
            raise ValueError(f"Header non valido nella prima linea del file .nnet: '{lines[0]}'") from exc
        if len(header) < 2:
            raise ValueError(f"Header incompleto: attesi almeno num_layers e input_size, ottenuto {header}")
        num_layers = header[0]
        input_size = header[1]
        
        # Line 1: Layer sizes (ignored here, inferred from weights)
        # Line 2: Symmetric flag (unused)
        
        # Line 3: Input minimums
        self.min_inputs = self._to_float_vector(lines[3].split(","), "min_inputs")
        
        # Line 4: Input maximums
        self.max_inputs = self._to_float_vector(lines[4].split(","), "max_inputs")
        
        # Line 5: Means for normalization
        self.means = self._to_float_vector(lines[5].split(","), "means")
        
        # Line 6: Ranges for normalization
        self.ranges = self._to_float_vector(lines[6].split(","), "ranges")

        # Load weights and biases
        current_line = 7
        for i in range(num_layers):
            # Dimensions for the current layer would be read here; kept for completeness
            # In the ACAS Xu schema layers are sequential. We assume the standard
            # structure (5 -> 50 -> ... -> 5) and parse weights and biases below.
            pass
        
        # Robust pass to extract weight matrices and bias vectors
        all_floats = []
        with open(self.filename, 'r') as f:
            for line in f:
                if line.strip().startswith("//") or not line.strip(): continue
                parts = line.split(",")[:-1] # Drop trailing empty if comma at end
                if not parts: parts = line.split(",") # Fallback
                try:
                    vals = [float(x) for x in parts if x.strip() != '']
                    all_floats.extend(vals)
                except:
                    pass
        
        # ACAS Xu networks have fixed structure: 5 inputs, 6 hidden layers of 50, 5 outputs
        layer_sizes = [5, 50, 50, 50, 50, 50, 50, 5]
        
        # Skip header metadata, then read weights/biases in order
        
        idx = 0
        # Reload only weights and biases, skipping the first 7 metadata lines
        raw_data = lines[7:] 
        
        val_idx = 0
        flat_data = []
        for l in raw_data:
            flat_data.extend([float(x) for x in l.split(",") if x.strip()])
            
        idx = 0
        for i in range(len(layer_sizes) - 1):
            rows = layer_sizes[i+1]
            cols = layer_sizes[i]
            
            # Weights (matrix rows x cols)
            w_size = rows * cols
            w = np.array(flat_data[idx : idx + w_size]).reshape(rows, cols)
            self.weights.append(w)
            idx += w_size
            
            # Bias (vector rows)
            b_size = rows
            b = np.array(flat_data[idx : idx + b_size])
            self.biases.append(b)
            idx += b_size

    def predict(self, inputs):
        """
        inputs: numpy array (N, 5) not normalized
        """
        # 1. Normalization (critical for ACAS Xu)
        # Standardize inputs: (x - mean) / range
        # Last value of means/ranges is for output; use only the first 5
        inp_norm = (inputs - self.means[:-1]) / self.ranges[:-1]
        
        # 2. Forward pass
        activation = inp_norm.T # Transpose for matrix multiplication (5, N)
        
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            # W is (50, 5), activation is (5, N) -> z is (50, N)
            z = np.dot(W, activation) + b[:, np.newaxis]
            
            # Apply ReLU to all but last layer
            if i < len(self.weights) - 1:
                activation = np.maximum(0, z) # ReLU
            else:
                activation = z # Last layer linear
                
        return activation.T # Return (N, 5)

    def generate_dataset(self, num_samples=10000):
        # Generate random inputs uniformly within min/max from the header
        
        # Create random matrix (N, 5)
        random_inputs = np.random.uniform(0, 1, (num_samples, 5))
        
        # Scale to real ranges: min + rand * (max - min)
        delta = np.subtract(self.max_inputs, self.min_inputs, dtype=float)
        real_inputs = self.min_inputs + random_inputs * delta
        
        # Get labels from the oracle
        print("Computing oracle outputs...")
        labels = self.predict(real_inputs)
        
        return real_inputs, labels  



# --- EXECUTION ---
if __name__ == "__main__":
    default_output_dir = os.path.join(os.getcwd(), "ACAS_Xu_datasets")
    parser = argparse.ArgumentParser(description="Generate dataset from ACAS Xu .nnet network")

    parser.add_argument(
        'input_file', 
        nargs='?',
        default=None,
        type=str, 
        help='.nnet file path'
        )
    
    parser.add_argument(
        '-d', '--directory',
        type = str,
        default=default_output_dir,
        help=f"Output directory (default: {default_output_dir})",
        metavar='DIR'
    )

    parser.add_argument(
        '-u', '--unified',
        type = str,
        default= None,
        metavar='DIR',
        help='If set, generates a unified Super-Dataset from all .nnet files in specified directory'
    )
    
    args = parser.parse_args()

    nnet_path = args.input_file
    output_dir = args.directory
    unified_dir = args.unified


    if nnet_path is None and unified_dir is None:
        print("Error: You must specify either a single input file OR use -u with a directory.")
        parser.print_usage()
        raise SystemExit(1)

    os.makedirs(output_dir, exist_ok=True)

    if unified_dir is not None:
        if not os.path.isdir(unified_dir):
            print(f"Error: With -u flag, unified_dir must be a directory. Got: {unified_dir}")
            raise SystemExit(1)
            
        print(f"Generating UNIFIED dataset from folder: {unified_dir}")
        
        # Generiamo meno campioni per file per non esplodere la RAM
        # 45 file * 5000 = 225.000 campioni totali
        SAMPLES_PER_FILE = 5000 
        
        try:
            X_all, Y_all = generate_unified_dataset(unified_dir, samples_per_file=SAMPLES_PER_FILE)
            
            print(f"Unified Dataset Shape: Input {X_all.shape} (7D), Output {Y_all.shape}")
            
            # Split
            X_train, X_test, Y_train, Y_test = train_test_split(
                X_all, Y_all, test_size=0.2, random_state=42
            )
            
            # Salvataggio
            dataset_dir = "ACAS_Xu_Unified_Dataset"
            out_path = os.path.join(output_dir, dataset_dir)
            os.makedirs(out_path, exist_ok=True)
            
            np.savez(os.path.join(out_path, "acas_xu_unified_train.npz"), X=X_train, Y=Y_train)
            np.savez(os.path.join(out_path, "acas_xu_unified_test.npz"), X=X_test, Y=Y_test)
            print(f"Super-Dataset salvato in {out_path}")
            
        except Exception as e:
            print(f"Errore durante la generazione unificata: {e}")
            SystemExit(1)

        # --- LOGICA SINGOLA (Legacy) ---
    else:
        if not os.path.isfile(nnet_path):
            print(f"Error: File {nnet_path} not found (did you mean to use -u for directory?)")
            SystemExit(1)
            
        print(f"Generating single dataset from file: {nnet_path}")
        
        N_TOTAL = 100000 
        oracle = ACASXuOracle(nnet_path)
        X_all, Y_all = oracle.generate_dataset(N_TOTAL)
        
        X_train, X_test, Y_train, Y_test = train_test_split(
            X_all, Y_all, test_size=0.2, random_state=42
        )
        
        print(f"Train: {X_train.shape}, Test: {X_test.shape}")
        
        base_name = os.path.splitext(os.path.basename(nnet_path))[0]
        out_filename = f"{base_name}"
        dataset_dir = f"{base_name}_dataset"
        out_path = os.path.join(output_dir, dataset_dir)
        os.makedirs(out_path, exist_ok=True)

        np.savez(os.path.join(out_path, f"{base_name}_train.npz"), X=X_train, Y=Y_train)
        np.savez(os.path.join(out_path, f"{base_name}_test.npz"), X=X_test, Y=Y_test)
        print(f"Dataset singolo salvato in {out_path}")
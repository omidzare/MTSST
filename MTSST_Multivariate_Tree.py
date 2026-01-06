import numpy as np
import pandas as pd
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
import matplotlib.pyplot as plt
import graphviz
import os

# --- Data Loading ---

def load_ts_file(file_path):
    """
    Loads a .ts file (sktime/aeon format) into a pandas DataFrame (X) and a numpy array (y).
    This parser handles basic multivariate .ts files where dimensions are separated by ':' 
    and values are comma-separated.
    """
    data = []
    labels = []
    metadata = {}
    
    print(f"Reading file: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r') as f:
        # Read header
        line = f.readline()
        while line:
            line = line.strip()
            if not line or line.startswith("#"):
                line = f.readline()
                continue
                
            if line.startswith("@data"):
                break
                
            if line.startswith("@"):
                parts = line.split(" ")
                key = parts[0][1:]
                value = " ".join(parts[1:])
                metadata[key] = value
                
            line = f.readline()
            
        # Read data
        line = f.readline()
        while line:
            line = line.strip()
            if not line:
                line = f.readline()
                continue
                
            # Split class label from data
            # Format is typically: dim1_vals:dim2_vals:dim3_vals:class_label
            # OR dim1_vals,dim2_vals,class_label ? 
            # Standard .ts for multivariate is dims separated by ':' and values by ','
            # The class label is usually at the end.
            
            parts = line.split(":")
            
            # Check if last part is just the label (colon separated) or data+label (comma separated)
            last_part = parts[-1]
            if "," in last_part:
                # Comma separated label in last dimension string
                last_part_split = last_part.split(",")
                class_label = last_part_split[-1]
                parts[-1] = ",".join(last_part_split[:-1])
            else:
                # Colon separated label
                class_label = last_part
                parts = parts[:-1]
                
            # Process each dimension
            series_data = []
            for dim_str in parts:
                # each dim_str is a CSV of values
                series_data.append(np.array([float(x) for x in dim_str.split(",") if x.strip()]))
            
            # series_data is list of arrays (n_dims, n_timepoints)
            # We want to store this. 
            # If we convert to numpy array here, it must be uniform length.
            data.append(np.array(series_data).T) # Transpose to (n_timepoints, n_dims) for fastdtw
            
            labels.append(class_label)
            
            line = f.readline()
            
    # Convert to object array to handle potential variable lengths, though fastdtw handles it.
    X = np.array(data, dtype=object)
    y = np.array(labels)
    
    # Try to convert labels to integers if possible, or floats
    try:
        y = y.astype(float).astype(int)
    except ValueError:
        pass # keep as strings if they are labels like 'class1'
        
    return X, y

# --- Model Definitions ---

class Node:
    def __init__(self, witness, threshold, steps, class_distribution):
        self.witness = witness #stands for witness time series
        self.threshold = threshold
        self.steps = steps
        self.class_distribution = class_distribution
        self.TrueChild = None
        self.FalseChild = None

#Function for calculating the loss using Gini impurity
def Gini(group1, group2):
    _, counts1 = np.unique(group1, return_counts=True)
    _, counts2 = np.unique(group2, return_counts=True)
    gini1 = 1 - sum((count / len(group1)) ** 2 for count in counts1)
    gini2 = 1 - sum((count / len(group2)) ** 2 for count in counts2)
    return gini1 + gini2


#Function for calculating the loss using entropy
def InformationGain(group1, group2):
    # Compute the entropy of a group
    def entropy(group):
        if len(group) == 0: return 0
        _, counts = np.unique(group, return_counts=True)
        probs = counts / len(group)
        return -sum(p * np.log2(p) for p in probs)
    
    # Compute the information gain
    entropy_before = entropy(list(group1) + list(group2))
    entropy_after = (len(group1) * entropy(group1) + len(group2) * entropy(group2)) / (len(group1) + len(group2))
    return entropy_before - entropy_after

# Main function for training the tree
def TS_Step_Tree_Train(X, Y, maxstep, stepsize, maxd, mins, exclusive_step=True):
    def frequencies(Y):
        unique, counts = np.unique(Y, return_counts=True)
        return dict(zip(unique, counts))
    
    def Compute_Distance_Matrix(X_subset):
        # X_subset should be (n_samples, current_steps, n_dims)
        distance_matrix = np.zeros((len(X_subset), len(X_subset)))
        for i in range(len(X_subset)):
            for j in range(i, len(X_subset)):
                # fastdtw expects (n_points, n_dims)
                # Our load_ts_file returns (n_points, n_dims) arrays in X list.
                # But X_subset slicing might need care.
                
                # If X contains arrays of shape (n_points, n_dims)
                ts_i = X_subset[i]
                ts_j = X_subset[j]
                
                # Handle 1D case if necessary (fastdtw expects 1D or 2D)
                if ts_i.ndim == 1:
                    ts_i = ts_i.reshape(-1, 1)
                    ts_j = ts_j.reshape(-1, 1)
                    
                distance = fastdtw(ts_i, ts_j, dist=euclidean)[0]
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
        return distance_matrix
    
    def Find_Witness(DM, Y):
        max_loss = 0
        witness, wth = None, None

        for i in range(len(X)):
            value, threshold = best_cut(i, Y, DM, InformationGain)
            if value > max_loss:
                max_loss = value
                witness = X[i]
                wth = threshold

        return max_loss, witness, wth
    
    def best_cut(idx, Y, DM, Loss):
        pairs = [(DM[j, idx], Y[j]) for j in range(len(Y))]
        pairs.sort(key=lambda x: x[0])
        # Avoid division by zero by checking if slice is valid
        loss_values = []
        for j in range(1, len(Y)):
             loss_values.append(Loss([p[1] for p in pairs[:j]], [p[1] for p in pairs[j:]]))
        
        if not loss_values:
            return 0, 0

        j = np.argmax(loss_values)
        threshold = (pairs[j][0] + pairs[j+1][0])/2 # Fixed index from j-1 to j+1 logic (original was j-1?)
        # Original: threshold = (pairs[j][0] + pairs[j-1][0])/2  where j is index in loss_values (0 to len-2)
        # If j=0, pairs[j]=pairs[0], pairs[j-1] is last? No.
        # Range is 1 to len(Y). So j is index into loss_values corresponding to slit at j.
        # pairs[:j] is first j elements. pairs[j:] is rest.
        # Split point is between j-1 and j in pairs list?
        # Let's keep original logic but be careful about indices.
        
        # Original code:
        # loss_values = [Loss(..., ...) for j in range(1, len(Y))]
        # j = np.argmax(loss_values) 
        # threshold = (pairs[j][0] + pairs[j-1][0])/2
        
        # If argmax is 0 (first element of loss_values), it corresponds to j=1 in range.
        # So we want split at j=1. pairs[1] and pairs[0]. 
        # argmax returns 0. we need pairs[0+1] and pairs[0].
        # So it should be pairs[j_idx+1] and pairs[j_idx]? 
        # Let's assume original code logic: pairs is sorted.
        # If j comes from argmax of list of size N-1.
        # It maps to split index j+1.
        
        real_j = j + 1
        threshold = (pairs[real_j][0] + pairs[real_j-1][0])/2
        return loss_values[j], threshold
    
    # Check stopping criteria
    # min length check: 
    min_len = min(len(ts) for ts in X)
    
    if len(set(Y)) == 1 or maxd == 0 or len(X) < mins or min_len < stepsize:
        return Node(None, None, None, frequencies(Y))
    else:
        real_maxstep = min(maxstep, min_len)
        i = 1
        bestIg = 0
        best_witness, best_threshold, best_steps = None, None, None
        
        # X is list of arrays. Slicing X[:, ...] depends on X being numpy 2D array of values?
        # In multivariate case with object array of (time, dims), we can't slice X[:, 0:steps].
        # We need to slice each time series.
        
        while i * stepsize <= real_maxstep:
            # Create subset of time series trimmed to current step
            current_limit = i * stepsize
            
            # Slice each TS in X
            X_sliced = np.array([ts[:current_limit] for ts in X], dtype=object)
            
            DM = Compute_Distance_Matrix(X_sliced)
            Ig, witness, threshold = Find_Witness(DM, Y)
            if bestIg < Ig:
                # witness here is the full TS from X[i], we should probably store just the prefix?
                # The prediction code uses witness.reshape(-1,1) and slices x[0:node.steps].
                # So we should store the trimmed witness.
                bestIg = Ig
                best_witness = witness[:current_limit]
                best_threshold = threshold
                best_steps = current_limit
            i += 1
        
        if bestIg == 0:
            return Node(None, None, None, frequencies(Y))
        else:
            node = Node(best_witness, best_threshold, best_steps, frequencies(Y))
            
            # Split data
            # We need to compute distance of all X to best_witness (trimmed)
            TrueIndexes = []
            for k in range(len(X)):
                dist = fastdtw(best_witness, X[k][:best_steps], dist=euclidean)[0]
                if dist <= best_threshold:
                    TrueIndexes.append(k)
                    
            FalseIndexes = [k for k in range(len(X)) if k not in TrueIndexes]
            
            # Recursive calls
            # For TrueChild: truncate X?
            # Original: X[TrueIndexes, best_steps+1:]
            # This implies sliding window or advancing start time?
            # "prediction code: predict(node.TrueChild, x[0:node.steps])" -> wait, recursion on x?
            # Original predict:
            # if distance <= threshold: predict(node.TrueChild, x[0:node.steps]) -> likely TYPO in original notebook?
            # Usually we pass x[node.steps:] (cutting off used part) or keep full x?
            
            # Let's look at original notebook code for predict:
            # if distance <= node.threshold:
            #    return predict(node.TrueChild, x[0:node.steps])  <-- This looks suspicious. x[0:steps] is the part we just matched!
            #    Maybe it meant x[node.steps:]?
            
            # However, I should stick to the notebook's logic unless it's clearly broken.
            # But wait, step tree usually means we advance in time.
            
            # Let's look at Training recursion:
            # TrueChild = Train(X[TrueIndexes, best_steps+1:], ...) 
            # This slices columns. So it advances time.
            
            # So in Predict, we should pass the REMAINDER of the time series.
            # x[node.steps:] would be the remainder. 
            # Original notebook `x[0:node.steps]` essentially passes the SAME prefix? That would lead to infinite loop or reprocessing same data?
            # Ah, maybe it's `x[node.steps:]`?
            
            # Let's assume standard Shapelet/Step logic: consume prefix, pass suffix.
            # I will fix this to `x[node.steps:]` in predict based on the training slicing `best_steps+1:`.
            # Note: `best_steps+1` drops one point? usually `best_steps:`?
            
            X_true_next = np.array([X[k][best_steps:] for k in TrueIndexes], dtype=object)
            Y_true_next = Y[TrueIndexes]
            
            node.TrueChild = TS_Step_Tree_Train(X_true_next, Y_true_next, maxstep, stepsize, maxd-1, mins, exclusive_step)
            
            if exclusive_step:
                # exclusive_step means False branch uses SAME data (didn't match pattern, try another on same data)
                X_false_next = X[FalseIndexes] # No slicing
                Y_false_next = Y[FalseIndexes]
                node.FalseChild = TS_Step_Tree_Train(X_false_next, Y_false_next, maxstep, stepsize, maxd-1, mins, exclusive_step)
            else:
                # non-exclusive: advance time on False branch too?
                X_false_next = np.array([X[k][best_steps:] for k in FalseIndexes], dtype=object)
                Y_false_next = Y[FalseIndexes]
                node.FalseChild = TS_Step_Tree_Train(X_false_next, Y_false_next, maxstep, stepsize, maxd-1, mins, exclusive_step)
                
            return node

def predict(node, x):
    if node.witness is None:
        return max(node.class_distribution, key=node.class_distribution.get)
    
    # Check if x is long enough
    if len(x) < node.steps:
         return max(node.class_distribution, key=node.class_distribution.get)

    # Compute distance to witness
    # Witness is already (steps, dims). x should be sliced to (steps)
    dist = fastdtw(node.witness, x[:node.steps], dist=euclidean)[0]
    
    if dist <= node.threshold:
        if node.TrueChild is not None:
            # Advance time by node.steps
            return predict(node.TrueChild, x[node.steps:])
        else:
            return max(node.class_distribution, key=node.class_distribution.get)
    else:
        if node.FalseChild is not None:
            # If exclusive, don't advance time?
            # We need to know if the tree was trained exclusive or not? 
            # The structure doesn't store 'exclusive' flag.
            # But the logic implies: if we take False edge, do we consume steps?
            # In Training:
            # if exclusive_step: FalseChild gets X[:] (full)
            # else: FalseChild gets X[steps:]
            
            # We can't know for sure here without a flag in Node.
            # But typically for Shapelets/Step trees:
            # Match -> Consume -> Recurse True
            # No Match -> Try Next Shapelet at SAME position (for exclusive flow where we want to find *a* match)
            # OR No Match -> Advance (if we cover the timeline)
            
            # Since we can't change the Node class easily without breaking compatibility slightly,
            # let's assume we need to handle both or add a flag to Node.
            # I'll add `exclusive` flag to Node for safety if I can. 
            # Or just assume Exclusive for now based on notebook default `exclusive_step=True`.
            
            # HOWEVER, the notebook has `predict` function that does:
            # return predict(node.FalseChild, x[0:node.steps]) (Original code)
            # This passed the PREFIX again? That supports the "Same Position" theory.
            # But `x[0:steps]` limits it to just that prefix. It denies access to future data!
            # That seems wrong if FalseChild needs to search effectively.
            
            # Correct logic for Exclusive Step Tree:
            # If Match: Go True, Advance Time (consume prefix).
            # If No Match: Go False, Keep Time (retry current prefix with different witness).
            
            # So `predict(node.FalseChild, x)` (full x) would be correct for exclusive.
            # The original code `x[0:node.steps]` is very strange. 
            # It implies the False child only gets to look at the SAME window?
            
            # I will use `x` (current x) for False branch, assuming Exclusive logic which is standard.
            return predict(node.FalseChild, x) 
        else:
            return max(node.class_distribution, key=node.class_distribution.get)

def visualize_step_tree(node, dot=None):
    if dot is None:
        dot = graphviz.Digraph()

    if node.TrueChild is None and node.FalseChild is None:
        node_label = 'Leaf Node\n' + ('No Data' if not node.class_distribution else 'frequencies: ' + str(node.class_distribution))
    else:
        # Witness might be long, truncate for display
        w_str = str(node.witness.shape) if hasattr(node.witness, 'shape') else str(node.witness)
        node_label = 'witness: {}\nthreshold: {:.2f}\nsteps: {}\n'.format(w_str, node.threshold if node.threshold else 0, node.steps)
        node_label += 'No Data' if not node.class_distribution else 'frequencies: ' + str(node.class_distribution)

    dot.node(str(id(node)), label=node_label)

    if node.TrueChild is not None:
        dot.edge(str(id(node)), str(id(node.TrueChild)), label='True')
        visualize_step_tree(node.TrueChild, dot)

    if node.FalseChild is not None:
        dot.edge(str(id(node)), str(id(node.FalseChild)), label='False')
        visualize_step_tree(node.FalseChild, dot)

    return dot

# --- Main Execution ---

if __name__ == "__main__":
    # Parameters
    dataset_name = "Epilepsy"
    train_file = f'Multivariate_ts_dataset/{dataset_name}/{dataset_name}_TRAIN.ts'
    test_file = f'Multivariate_ts_dataset/{dataset_name}/{dataset_name}_TEST.ts'
    
    print(f"Loading {dataset_name} dataset...")
    if not os.path.exists(train_file):
        print(f"Error: Dataset not found at {train_file}")
        print("Ensure you are running this script from the MTSST_CODE directory.")
        exit(1)
        
    X_train, Y_train = load_ts_file(train_file)
    X_test, Y_test = load_ts_file(test_file)
    
    print(f"Train shape: {X_train.shape} samples (objects)")
    print(f"Test shape: {X_test.shape} samples (objects)")
    # Sample dimensions check
    print(f"Sample 0 shape: {X_train[0].shape} (timepoints, dims)")

    # Model Params
    maxstep = 50
    stepsize = 10
    maxd = 7
    mins = 2
    
    # --- Exclusive Step Tree ---
    print("\nTraining Exclusive Step Tree...")
    root_exclusive = TS_Step_Tree_Train(X_train, Y_train, maxstep, stepsize, maxd, mins, exclusive_step=True)
    
    # Predict
    print("Evaluating Exclusive Step Tree...")
    probabilities_exclusive = []
    predictions_exclusive = []
    
    # Identify positive class label (assuming 1 is positive, or max label)
    # The dataset labels might be 1, 2, 3, 4 etc.
    # For ROC, we need binary. Epilepsy is 4 classes?
    # Let's check classes.
    classes = np.unique(Y_train)
    print(f"Classes: {classes}")
    
    # If binary:
    if len(classes) == 2:
        pos_label = classes[1] # Assumes sorted
        neg_label = classes[0]
        
        for x in X_test:
            predicted_class = predict(root_exclusive, x)
            predictions_exclusive.append(predicted_class)
            # Prob as binary 0/1 for ROC
            probabilities_exclusive.append(1 if predicted_class == pos_label else 0)
            
        # Metrics
        print(classification_report(Y_test, predictions_exclusive))
        try:
            # Convert Y_test to binary 0/1
            Y_test_bin = [1 if y == pos_label else 0 for y in Y_test]
            auroc = roc_auc_score(Y_test_bin, probabilities_exclusive)
            print(f"AUROC (Exclusive): {auroc:.4f}")
        except:
            print("AUROC calculation failed (possibly multiclass).")
            
    else:
        # Multiclass support
        print("Multiclass dataset detected. Skipping ROC for now.")
        for x in X_test:
            predictions_exclusive.append(predict(root_exclusive, x))
        print(classification_report(Y_test, predictions_exclusive))

    # Visualize
    try:
        dot = visualize_step_tree(root_exclusive)
        out_file = f'tree_exclusive_{dataset_name}'
        dot.render(out_file, format='png', cleanup=True)
        print(f"Tree visualization saved to {out_file}.png")
    except Exception as e:
        print(f"Visualization failed: {e}")
        print("Make sure Graphviz is installed.")

    # --- Non-Exclusive Step Tree (Optional, can be uncommented) ---
    # print("\nTraining Non-Exclusive Step Tree...")
    # root_non_exclusive = TS_Step_Tree_Train(X_train, Y_train, maxstep, stepsize, maxd, mins, exclusive_step=False)
    # ... (evaluation code similar to above)

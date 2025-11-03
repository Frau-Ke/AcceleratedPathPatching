 
def circuit_size(CIRCUIT:dict) -> int:
    return sum([len(val) for val in CIRCUIT.values()])


def IoU_nodes(circuit1:dict, circuit2:dict):

    all_layers = set(circuit1.keys()) | set(circuit2.keys())
    
    total_intersection = 0
    total_union = 0
    
    for layer in all_layers:
        # Get the set of heads for the current layer in each dictionary (default to empty set if layer is absent)
        heads1 = set(circuit1.get(layer, []))
        heads2 = set(circuit2.get(layer, []))
        
        # Compute intersection and union for this layer
        intersection = heads1 & heads2
        union = heads1 | heads2
        
        total_intersection += len(intersection)
        total_union += len(union)
    
    # Avoid division by zero: if both dictionaries have no heads, define IoU as 1.
    if total_union == 0:
        return 1.0
    else:
        return total_intersection / total_union
    
def precision(circuit:dict, GT_circuit:dict):
    """The precision is TP / (TP + FN)

    Args:
        circuit (dict): tested circuit
        GT_circuit (dict): ground truth

    Returns:
        float: presicion
    """
    true_positive = get_intersection_num(circuit, GT_circuit)
    total_circuit = circuit_size(circuit)
    if total_circuit == 0:
        return 0
    else:
        return float(true_positive / total_circuit)


def TPR(circuit:dict, GT_circuit:dict):
    """The recall is TP / (TP + TN). Recall, sensitivity. 
    
    Args:
        circuit (dict): tested circuit
        GT_circuit (dict): ground truth

    Returns:
        float: TPR
    """
    # recall, sensitivity
    true_positive = get_intersection_num(circuit, GT_circuit)
    total_circuit = circuit_size(GT_circuit)
    if total_circuit == 0:
        return 0
    else:
        return float(true_positive / total_circuit)


def FPR(circuit:dict, GT_circuit:dict):
    true_positive = get_intersection_num(circuit, GT_circuit)
    total_circuit = circuit_size(circuit)
    if total_circuit == 0:
        return 0
    else:
        return (total_circuit - true_positive) / total_circuit
    

def merge_circuits(circuit1:dict, circuit2:dict) -> dict:
    """Union between two circuits

    Args:
        circuit1 (dict): circuit1
        circuit2 (dict): circuit2

    Returns:
        dict: Union of circuit1 and circuit2
    """
    
    new_circuit = {}
    all_layers = set(circuit1.keys()) | set(circuit2.keys())
    for layer in all_layers:
        heads1 = set(circuit1.get(layer, []))
        heads2 = set(circuit2.get(layer, []))
        union = heads1 | heads2
        new_circuit[layer] = union
    return new_circuit


def get_difference(circuit1:dict, circuit2:dict) -> dict:
    """Get the difference of circuit1 from circuit2 returned as a new circuit

    Args:
        circuit1 (dict): dict of circuit1
        circuit (dict): dict of circuit2

    Returns:
        dict: the difference between circuit1 and circuit2
    """
    
    all_layers = set(circuit1.keys()) | set(circuit2.keys())
    difference = {}
    for layer in all_layers:
        heads1 = set(circuit1.get(layer, []))
        heads2 = set(circuit2.get(layer, []))
        diff = heads1.difference(heads2)
        if len(diff) > 0:
            
            difference[layer] = list(diff)
    
    return difference


def intersect_circuits(circuit1:dict, circuit2:dict) -> dict:
    """Get the intersection of two circuits as a new circuit

    Args:
        circuit1 (dict): circuit1
        circuit2 (dict): circuit2
    Returns: new_circuit (dict): intersection between circuit1 and circuit2
    """
    new_circuit = {}
    all_layers = set(circuit1.keys()) | set(circuit2.keys())

    for layer in all_layers:
        heads1 = set(circuit1.get(layer, []))
        heads2 = set(circuit2.get(layer, []))
        new_circuit[layer] = heads1.intersection(heads2)
    return new_circuit


def get_intersection_num(circuit1:dict, circuit2:dict) -> int:
    """Count the number of intersecting nodes

    Args:
        circuit1 (dict): circuit1
        circuit2 (dict): circuit2

    Returns:
        int: number of intersections
    """
    all_layers = set(circuit1.keys()) | set(circuit2.keys())
    
    total_intersection = 0
    
    for layer in all_layers:
        # Get the set of heads for the current layer in each dictionary (default to empty set if layer is absent)
        heads1 = set(circuit1.get(layer, []))
        heads2 = set(circuit2.get(layer, []))
        
        # Compute intersection and union for this layer
        intersection = heads1 & heads2
        total_intersection += len(intersection)
    return total_intersection


def get_union_num(circuit1:dict, circuit2:dict) -> int:
    """Count union of both circuits

    Args:
        circuit1 (dict): circuit1
        circuit2 (dict): circuit2

    Returns:
        int: size of circuit union
    """

    all_layers = set(circuit1.keys()) | set(circuit2.keys())
    
    total_union = 0
    
    for layer in all_layers:
        # Get the set of heads for the current layer in each dictionary (default to empty set if layer is absent)
        heads1 = set(circuit1.get(layer, []))
        heads2 = set(circuit2.get(layer, []))
        
        # Compute intersection and union for this layer
        union = heads1 | heads2
        total_union += len(union)
    return total_union

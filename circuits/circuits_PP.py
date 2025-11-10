def choose_PP_circuit(task, model_name):
    return {}

def get_circuit_name(task):
    if task == "IOI":
        name = "IOI Circuit"
    elif task == "GenderedPronouns":
        name = "Gendered Pronouns Circuit"   
    elif task == "GreaterThan":
        name = "Greater Than Circuit"
    elif  task == "Induction":
        name = "Induction Circuit"
    elif task == "Docstring":
        name = "Docstring"
    else:
        raise Exception(f"for the task {task} there is no circuit name defined.")
    
    return name

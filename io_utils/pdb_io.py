import os


def list_pdb_files(pdb_dir):
    pdb_files = [f for f in os.listdir(pdb_dir) if f.endswith('.pdb')]
    
    if not pdb_files:
        print("No PDB files found in the directory.")
        return []
    
    print("Select PDB files to associate:")
    print("1 - All")
    
    for i, pdb_file in enumerate(pdb_files, start=2):
        print(f"{i} - {pdb_file}")
    
    return pdb_files


def get_user_selection(pdb_files, pdb_dir):
    selection = input("\nEnter the numbers of the proteins to associate (comma-separated) or '1' to select all: ")
    
    try:
        reference_graph_indice = int(input("Enter the number of your reference graph or '1' to let system choose: "))
        selected_numbers = [int(num.strip()) for num in selection.split(",")]
        
        if reference_graph_indice not in selected_numbers and reference_graph_indice != 1:
            raise ValueError("The reference graph must be one of the selected proteins.")
    except ValueError as e:
        print("Invalid input for reference graph. Please enter a valid number.")
        return get_user_selection(pdb_files, pdb_dir)
    
    if reference_graph_indice == 1:
        reference_graph = None
    else:
        try:
            reference_graph = pdb_files[reference_graph_indice - 2]
            reference_graph = os.path.join(pdb_dir, reference_graph)
        except IndexError:
            print("Reference graph number out of range.")
            return get_user_selection(pdb_files, pdb_dir)
    
    if 1 in selected_numbers:
        selected_files = [[os.path.join(pdb_dir, pdb), pdb] for pdb in pdb_files]
    else:
        selected_files = []
        for i in selected_numbers:
            try:
                selected_files.append([os.path.join(pdb_dir, pdb_files[i - 2]), pdb_files[i - 2]])
            except IndexError:
                print(f"Number {i} is out of range. Skipping.")
    
    return selected_files, reference_graph

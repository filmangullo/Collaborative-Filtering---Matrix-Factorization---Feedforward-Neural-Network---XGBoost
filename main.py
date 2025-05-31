import subprocess

def main():
    # PRINT TITLE
    width_title = 80
    print("-" * width_title)
    print("OPTIMIZATION OF HYBRID-BASED COLLABORATIVE FILTERING".center(width_title))
    print("USING".center(width_title))
    print("MATRIX FACTORIZATION, FEEDFORWARD NEURAL NETWORK, AND XGBOOST".center(width_title))
    print("TO IMPROVE RECOMMENDATIONS".center(width_title))
    print("-" * width_title)

    print("Select Dataset:")
    print("1. Dummny")
    print("2. Film (MovieLens)")
    print("3. Hotel (PT. XYZ)")
    dataset_choice = input("Choose dataset (1, 2 or 3): ")

    if dataset_choice == '1':
        dataset = "dummy"
    elif dataset_choice == '2':
        dataset = "movie"
    elif dataset_choice == '3':
        dataset = "hotel"
    else:
        print("Invalid dataset selected.")
        return
    

    print("Select the program you want to run:")
    print("1. Rating Prediction: Matrix Factorization and Feedforward Neural Network")
    print("2. Generate : Handcrafted Features")
    print("3. Do Recommendation : XGBoost")
    pilihan = input("Please select one of the following options: 1, 2, or 3: ")

    if pilihan == '1':
        subprocess.run(["python", "program_mf_ffnn.py", dataset])
    elif pilihan == '2':
        subprocess.run(["python", "program_hf.py"])
    elif pilihan == '3':
        subprocess.run(["python", "program_xg.py"])
    else:
        print("Invalid selection.")

if __name__ == "__main__":
    main()
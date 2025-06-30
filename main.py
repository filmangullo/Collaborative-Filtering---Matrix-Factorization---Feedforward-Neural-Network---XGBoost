import subprocess
import sys

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
    if isinstance(dataset_choice, int):
        dataset_choice = str(dataset_choice)
    else:
        dataset_choice = dataset_choice.strip()

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
    program_choice = str(input("Please select one of the following options: 1, 2, or 3: ")).strip()

    if program_choice == '1':
        subprocess.call([sys.executable, "program_mf_ffnn.py", dataset])
    elif program_choice == '2':
        subprocess.call([sys.executable, "program_hf_mf_ffnn.py", dataset])
    elif program_choice == '3':
        print("\nLaunching Streamlit app for XGBoost recommendation...")
        subprocess.call(["streamlit", "run", "program_xg.py", dataset])
    else:
        print("Invalid selection.")

if __name__ == "__main__":
    main()
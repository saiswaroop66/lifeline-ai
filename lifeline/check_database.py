

from database import get_counts, initialize_database

def main():
    print("\nLIFELINE DATABASE CHECK")
    print("-" * 35)

    initialize_database()
    counts = get_counts()

    print(f"Blood banks:      {counts['blood_banks']}")
    print(f"Inventory rows:   {counts['inventory']}")
    print(f"Component status: {counts['component_status']}")

    print("-" * 35)
    print("Database verification completed!")

if __name__ == "__main__":
    main()

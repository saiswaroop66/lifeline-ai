
print("LifeLine database check started!")

from check_database import get_counts, initialize_database

initialize_database()
counts = get_counts()

print("\nLIFELINE DATABASE CHECK")
print("-" * 35)
print("Blood banks:", counts["blood_banks"])
print("Inventory rows:", counts["inventory"])
print("Component status:", counts["component_status"])
print("-" * 35)
print("Database verification completed!")

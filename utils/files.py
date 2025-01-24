from dotenv import load_dotenv
import os

def write_to_env_file(key: str, value: str, env_path=".env"):
    """Write a key-value pair to the .env file."""
    # Check if the .env file exists
    if not os.path.exists(env_path):
        open(env_path, 'w').close()  # Create the file if it doesn't exist

    # Read the existing .env content
    with open(env_path, 'r') as file:
        lines = file.readlines()

    # Delete the existing key if it exists
    lines = [line for line in lines if not line.startswith(key)]
    
    # Add the new key-value pair
    lines.append(f"{key}={value}\n")

    # Write the updated content back to the .env file
    with open(env_path, 'w') as file:
        file.writelines(lines)
        
        
if __name__ == "__main__":
    write_to_env_file("TARGET_SIZE", f"(512, 512)")
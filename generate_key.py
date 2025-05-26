import os

key = os.urandom(24).hex()

with open(".env", "w") as f:
    f.write(f"SECRET_KEY={key}")

print("✅ SECRET_KEY berhasil dibuat dan disimpan di file .env")

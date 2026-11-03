
import requests

usernames = ["ayanatiq01-arch", "openai", "google"]

for username in usernames:
    url = f"https://api.github.com/users/{username}"

    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()

        print("Login:", data["login"])
        print("Public Repository Count:", data["public_repos"])
        print("-----------------------------")

    else:
        print("Error fetching data for:", username)
        print("Status Code:", response.status_code)

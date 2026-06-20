from git import Repo
import os

def clone_repository(repo_url):

    repo_name = (
        repo_url.rstrip("/")
        .split("/")[-1]
        .replace(".git", "")
        )

    repo_path = os.path.join(
        "data",
        "repositories",
        repo_name
    )

    if not os.path.exists(repo_path):
        Repo.clone_from(repo_url, repo_path)
        print("Repository cloned")
    else:
        print("Repository already exists")

    return repo_path
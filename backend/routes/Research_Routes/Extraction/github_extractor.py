import httpx

async def extract_github_content(repo_url):
    try:
        parts = repo_url.replace("https://github.com/", "").split("/")
        owner, repo = parts[0], parts[1]

        api_url = f"https://api.github.com/repos/{owner}/{repo}/readme"

        async with httpx.AsyncClient() as client:
            res = await client.get(api_url, headers={"Accept": "application/vnd.github.v3.raw"})

        if res.status_code == 200:
            return res.text
        else:
            return ""
    except:
        return ""
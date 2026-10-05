#!/usr/bin/env bash
# Helper script to create and push this repository to GitHub
set -euo pipefail

TOKEN="${1:-${GITHUB_TOKEN:-}}"
USERNAME="${2:-Novasaki}"
REPO_NAME="${3:-grug-speech-reasoning}"

if [ -z "$TOKEN" ]; then
    echo "Usage: $0 <github_personal_access_token> [github_username] [repo_name]"
    echo "Or set the GITHUB_TOKEN environment variable."
    exit 1
fi

echo "Configuring GitHub remote for $USERNAME/$REPO_NAME..."

# Test authentication via GitHub API
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: token $TOKEN" https://api.github.com/user)
if [ "$HTTP_STATUS" != "200" ]; then
    echo "Error: GitHub token validation failed (HTTP status $HTTP_STATUS). Check your token permissions."
    exit 1
fi

echo "Authenticated with GitHub successfully."

# Create repository if it doesn't already exist
curl -s -H "Authorization: token $TOKEN" \
     -H "Accept: application/vnd.github.v3+json" \
     https://api.github.com/user/repos \
     -d "{\"name\":\"$REPO_NAME\",\"description\":\"Grug Speech Reasoning Distillation Pipeline, Multi-Field Dataset & Cross-Model Benchmarks\",\"private\":false}" > /dev/null || true

# Set remote and push
git remote remove origin 2>/dev/null || true
git remote add origin "https://${TOKEN}@github.com/${USERNAME}/${REPO_NAME}.git"

echo "Pushing main branch to https://github.com/${USERNAME}/${REPO_NAME}..."
git push -u origin main --force

# Clean up remote URL to avoid storing token in plain text .git/config
git remote set-url origin "https://github.com/${USERNAME}/${REPO_NAME}.git"

echo "Successfully pushed to https://github.com/${USERNAME}/${REPO_NAME}!"

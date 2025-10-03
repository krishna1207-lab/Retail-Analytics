#!/bin/bash

# Script to push the retail analytics dashboard to GitHub
# This will clean the existing repository and push new content

echo "🚀 Pushing retail analytics dashboard to GitHub..."

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "❌ Not in a git repository. Please run this from the project directory."
    exit 1
fi

# Add all files
echo "📁 Adding all files..."
git add .

# Commit changes
echo "💾 Committing changes..."
git commit -m "Complete retail analytics dashboard with Streamlit and FastAPI

- 7 main analysis tabs with interactive visualizations
- Customer segmentation and RFM analysis
- Profitability and seasonal trend analysis
- Payment method and category insights
- Campaign simulation capabilities
- FastAPI backend integration
- Pre-trained ML models included"

# Force push to replace existing content
echo "🔄 Force pushing to GitHub (this will replace existing content)..."
git push -f origin main

if [ $? -eq 0 ]; then
    echo "✅ Successfully pushed to GitHub!"
    echo "🌐 Repository: https://github.com/krishna1207-lab/retail-analytics-dashboard"
    echo "📊 Dashboard can be deployed on Streamlit Cloud: https://share.streamlit.io"
else
    echo "❌ Failed to push to GitHub. Please check your authentication."
    echo "💡 You may need to:"
    echo "   1. Set up a Personal Access Token"
    echo "   2. Configure Git credentials"
    echo "   3. Or use SSH keys"
fi

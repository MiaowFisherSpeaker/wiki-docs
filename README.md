# Wiki Docs

A Docsify-powered documentation site deployed on GitHub Pages.

## 🚀 How to Publish Your Repository to GitHub Pages

This repository is already configured for GitHub Pages deployment. Follow these steps to publish your documentation:

### Step 1: Enable GitHub Pages in Repository Settings

1. Go to your repository on GitHub: `https://github.com/MiaowFisherSpeaker/wiki-docs`
2. Click on **Settings** tab
3. Scroll down to **Pages** section in the left sidebar
4. Under **Source**, select **GitHub Actions**
5. Save the settings

### Step 2: Push Changes to Trigger Deployment

The repository includes a GitHub Actions workflow (`.github/workflows/deploy.yml`) that automatically deploys your documentation when you:

- Push changes to the `main` or `master` branch
- Create a pull request to these branches
- Manually trigger the workflow from the Actions tab

### Step 3: Access Your Published Site

Once deployed, your documentation will be available at:
- **Custom Domain**: https://wiki.miaowfisherspeaker.top (configured via `docs/CNAME`)
- **GitHub Pages URL**: https://miaowfisherspeaker.github.io/wiki-docs

## 📁 Repository Structure

```
wiki-docs/
├── .github/
│   └── workflows/
│       └── deploy.yml          # GitHub Actions workflow for deployment
├── docs/                       # Documentation source files
│   ├── index.html             # Docsify entry point
│   ├── README.md              # Home page content
│   ├── _sidebar.md            # Navigation sidebar
│   ├── _navbar.md             # Top navigation
│   ├── .nojekyll              # Disables Jekyll processing
│   ├── CNAME                  # Custom domain configuration
│   ├── sw.js                  # Service worker
│   └── data_science/          # Content directories
│       └── ...
└── README.md                  # This file
```

## ✏️ Editing Your Documentation

### Adding New Pages

1. Create new `.md` files in the `docs/` directory or subdirectories
2. Update `docs/_sidebar.md` to include links to your new pages
3. Commit and push your changes

### Updating Content

1. Edit existing `.md` files in the `docs/` directory
2. Commit and push your changes
3. The site will automatically update via GitHub Actions

### Customizing the Site

- **Title & Description**: Edit `docs/index.html`
- **Navigation**: Edit `docs/_navbar.md` and `docs/_sidebar.md`
- **Home Page**: Edit `docs/README.md`
- **Domain**: Update `docs/CNAME` for custom domain

## 🔧 Local Development

To run the documentation site locally:

1. Install a local web server (e.g., Python's built-in server):
   ```bash
   cd docs
   python -m http.server 3000
   ```

2. Open your browser and navigate to `http://localhost:3000`

## 🎯 Features

- **Docsify**: Fast, lightweight documentation framework
- **GitHub Actions**: Automated deployment on every push
- **Custom Domain**: Configured for `wiki.miaowfisherspeaker.top`
- **Math Support**: KaTeX integration for mathematical formulas
- **Service Worker**: Offline support and improved performance
- **Responsive Design**: Mobile-friendly documentation

## 📚 Docsify Documentation

For more information about customizing your Docsify site, visit:
- [Docsify Documentation](https://docsify.js.org/#/)
- [Docsify Themes](https://docsify.js.org/#/themes)
- [Docsify Plugins](https://docsify.js.org/#/plugins)

## 🚨 Troubleshooting

### Deployment Issues

1. **Check Actions Tab**: Go to the Actions tab in your repository to see if the workflow is running
2. **Verify Settings**: Ensure GitHub Pages is configured to use "GitHub Actions" as the source
3. **Check Permissions**: Make sure the workflow has the necessary permissions (already configured)

### Custom Domain Issues

1. **DNS Configuration**: Ensure your domain's DNS is pointing to GitHub Pages
2. **CNAME File**: Verify `docs/CNAME` contains your domain name
3. **SSL Certificate**: GitHub will automatically provision SSL for custom domains

### Content Not Updating

1. **Clear Cache**: Try hard refresh (Ctrl+F5) or clear browser cache
2. **Check Workflow**: Verify the GitHub Actions workflow completed successfully
3. **Wait**: Changes may take a few minutes to propagate

## 📞 Support

If you encounter issues:
1. Check the [GitHub Pages documentation](https://docs.github.com/en/pages)
2. Review the [GitHub Actions documentation](https://docs.github.com/en/actions)
3. Check the repository's Issues tab for similar problems
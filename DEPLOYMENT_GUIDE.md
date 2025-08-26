# 🚀 QUICK DEPLOYMENT GUIDE

## 1️⃣ GitHub Setup (2 minutes)

### Create GitHub Repository:
1. Go to: https://github.com/new
2. Repository name: `kidney-stone-detection`
3. Description: `AI-powered kidney stone detection using CNN+SVM ensemble`
4. Set to Public
5. Click "Create repository"

### Push Your Code:
```bash
git remote set-url origin https://github.com/Gaurav050Sharma/kidney-stone-detection.git
git push -u origin master
```

## 2️⃣ Hugging Face Deployment (3 minutes)

### Create Space:
1. Go to: https://huggingface.co/new-space
2. Space name: `kidney-stone-detection`
3. License: `MIT`
4. SDK: `Gradio`
5. Hardware: `CPU basic`
6. Click "Create Space"

### Upload Files:
Upload these files to your Hugging Face Space:
- `app.py` (main interface)
- `model.py` (AI models)
- `utils.py` (helpers)
- `requirements_hf.txt` → rename to `requirements.txt`
- `README_HF.md` → rename to `README.md`
- `models/kidney_stone_cnn.h5`
- `models/kidney_stone_svm.pkl`

## 3️⃣ Your Live URLs:

🔗 **GitHub**: https://github.com/Gaurav050Sharma/kidney-stone-detection
🚀 **Live App**: https://huggingface.co/spaces/Gaurav050Sharma/kidney-stone-detection

## 🎉 That's it! Your AI is live and deployed!

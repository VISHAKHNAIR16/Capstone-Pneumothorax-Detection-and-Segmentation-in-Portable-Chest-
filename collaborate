## Getting Started: How to Collaborate

### 1. Clone the Repository

To start working on the project, clone the GitHub repository to your local machine:


git clone https://github.com/VISHAKHNAIR16/Capstone-Pneumothorax-Detection-and-Segmentation-in-Portable-Chest-.git

This copies the entire project so you can work locally.

---

### 2. Navigate Into the Project Folder


cd Capstone-Pneumothorax-Detection-and-Segmentation-in-Portable-Chest-

---

### 3. Create a New Branch for Your Work

Create and switch to a new branch before making changes. This keeps your work isolated and organized:


git checkout -b feature/your-feature-name

Replace `your-feature-name` with a descriptive name for your task.

---

### 4. Make Your Changes

Add or modify files for your task using your preferred editor.

---

### 5. Stage Your Changes for Commit

Add specific files:


git add filename.py

Or add all changes:


git add .

---

### 6. Commit Your Changes

Save your changes with a clear and concise commit message:


git commit -m "Brief description of your changes"

---

### 7. Sync with Remote Main Branch

Before pushing, make sure your local `main` branch is updated:


git checkout main
git pull origin main

Then rebase your feature branch on `main` (optional but recommended):


git checkout feature/your-feature-name
git rebase main

---

### 8. Push Your Branch to GitHub


git push origin feature/your-feature-name

---

### 9. Create a Pull Request (PR)

- Go to the GitHub repo page.
- Click "Compare & pull request" for your branch.
- Add a descriptive title and details.
- Assign reviewers and request reviews.
- Submit the PR.

---

### 10. Merge and Delete Branch

After PR approval and merge:

- Switch back to main:


git checkout main

- Pull latest changes:


git pull origin main

- Optionally delete your feature branch locally and remotely:


git branch -d feature/your-feature-name
git push origin --delete feature/your-feature-name

---

### Troubleshooting: Resolving Merge Conflicts

- Open conflicted files and fix issues manually.
- Stage the resolved files:


git add <conflicted-file>

- Continue rebase or commit:


git rebase --continue
or
git commit -m "Resolved conflicts"

param($BranchName="models_train", $RepoPath="C:\Users\sanjo\OneDrive\Attachments\Desktop\Coding Shit\ml-core-libraries-practice")

$errors = @()
for ($i = 1; $i -le 10; $i++) {
    try {
        Write-Host "=== Checkpoint $i ===" -ForegroundColor Cyan
        
        # Create a checkpoint file
        New-Item -Path "$RepoPath\checkpoint_$i.txt" -ItemType File -Force | Out-Null
        
        # Change to repo directory
        Set-Location $RepoPath
        
        # Add and commit
        git add -A
        $commitMsg = "Checkpoint $i: ML model development progress"
        git commit -m $commitMsg
        
        # Push to GitHub
        git push origin $BranchName
        
        Write-Host "Push $i complete`n" -ForegroundColor Green
    } catch {
        $errors += $_.Exception
        Write-Host "Error at checkpoint $i: $($_.Exception.Message)" -ForegroundColor Red
    }
}

if ($errors.Count -gt 0) {
    Write-Host "`nErrors encountered: $($errors.Count)" -ForegroundColor Red
    exit 1
} else {
    Write-Host "`nAll 10 checkpoints pushed successfully!" -ForegroundColor Green
}
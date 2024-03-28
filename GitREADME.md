

### 代码提交管理
```
git checkout -b x
# 在 feature-branch 分支上进行修改和提交
git add .
git commit -m "Add feature X"
git push origin x
```


### Git Rebase
* 在我们开发的过程中及时同步main分支的更新，从而避免后续代码更新过程中的矛盾信息

```
git checkout 当前分支
git rebase main 
解决矛盾冲突
git add  将解决后的文件标记为已解决
git rebase --continue 

```

### Git Merge
```
git checkout main 切换到main 分支
git pull origin main 拉去最新的main分支信息
git merge feature-branch    git merge 命令将你的特性分支合并到 main 分支上：
解决冲突 
git commit -m "CommitInfo" 
git push origin main 
```

###  分支管理

```
git branch x   新建分支
git checkout x  切换到分支

或者 git checkout -b x 切换并且新建分支

git branch -a 
git branch -r 
git branch 

git push origin --delete x 删除远程分支
git branch -D x 强制删除本地分支 
git branch -d x 安全删除本地分支  

```

### 分支管理
```
git log 
git checkout abcdef
git reset --hard abcdef

git diff <commit A> <commit B>  版本差异性代码

git diff 当前工作目录和暂存区的差异
git diff --cached 暂存区和最新commit的差异
git diff HEAD     工作区和最新commit的差异
git diff <commit> 当前工作区与之前某个特定 commit 之间的差异

```

### 👎哥的讲解
```
git add .
git commit -m "添加新功能"

git push origin feature-branch:feature-branch

```


https://juejin.cn/post/7038093620628422669

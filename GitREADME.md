## 目录
参考链接： 
https://juejin.cn/post/7038093620628422669


- [新建链接](#新建链接)
- [代码提交管理](#代码提交管理)
- [Git Rebase](#git-rebase)
- [Git Merge](#git-merge)
- [分支管理](#分支管理)
- [版本差异性](#版本差异性)
- [👎哥的讲解](#👎哥的讲解)
- [GitLab 和GitHub 同时的管理](#gitlab-和github-同时的管理)
### 新建链接
```
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin git@github.com:wangxinchao-bit/test.git
git push -u origin main 


git branch -M main：
这条命令将当前所在分支重命名为 main。通常情况下，Git 的默认分支名是 master，但为了避免含有带有种族主义和非平等意义的术语，许多项目已经决定使用更加中性的术语，比如 main。这条命令用于将默认分支名改为 main。

git push -u origin main：
这条命令将本地的 main 分支推送到远程的 origin 仓库，并将本地的 main 分支设置为远程 origin 仓库的默认分支。-u 选项用于建立本地分支和远程分支的关联，这样后续的推送和拉取操作就可以简化为 git push 和 git pull，而不需要指定分支名。
```


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

### 版本差异性
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


### GitLab 和GitHub 同时的管理
```
# 建立仓库

GitHub 上创建一个新的仓库：
首先，在 GitHub 上创建一个新的仓库，该仓库将用于存储你的代码。在创建仓库时，可以选择公开或私有，并记下 GitHub 仓库的 URL。

在 GitLab 上创建一个新的仓库：
同样地，在 GitLab 上创建一个新的仓库，该仓库也将用于存储你的代码。同样，可以选择公开或私有，并记下 GitLab 仓库的 URL。

# 关联仓库

将本地代码关联到 GitHub 仓库：
进入到你的本地代码目录，然后执行以下命令将本地代码关联到 GitHub 仓库：

git remote add github <github_repository_url>
将 <github_repository_url> 替换为你在 GitHub 上创建的仓库的 URL。例如：
git remote add github https://github.com/yourusername/github-repo.git
将本地代码关联到 GitLab 仓库：
同样地，执行以下命令将本地代码关联到 GitLab 仓库：


git remote add gitlab <gitlab_repository_url>
将 <gitlab_repository_url> 替换为你在 GitLab 上创建的仓库的 URL。例如：
git remote add gitlab https://gitlab.com/yourusername/gitlab-repo.git

# 代码推送

推送代码到 GitHub 和 GitLab：
一旦你将本地代码关联到了 GitHub 和 GitLab 仓库，你可以使用 git push 命令将代码推送到两个远程仓库：
git push github master
git push gitlab master

```



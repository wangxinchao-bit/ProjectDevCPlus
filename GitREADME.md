


### Git Rebase
* 在我们开发的过程中及时同步main分支的更新，从而避免后续代码更新过程中的矛盾信息

```
git checkout 当前分支
git rebase main 
解决矛盾冲突
git add  将解决后的文件标记为已解决
git rebase --continue 

```


https://juejin.cn/post/7038093620628422669

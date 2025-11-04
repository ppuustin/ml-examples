Intro
==============

This is a collection of simple s ml examples. 
A collection of examples for different purposes and then some.
Not organized.


```sh
git clone <repo>
git pull <repo> master  #update  ... git pull master
git chekcout master     #update in index or tree
git fetch origin master #pull w o merge
git rebase -i origin/master
```

```sh
git branch <branch-name>
git checkout <branch-name>
... blaah blaah
git git push --set-upstream origin <branch-name>
<pull request + review + push to main>
 
git checkout master
git pull
```

```sh
git status
git restore
git restore <file>      #rollback
git reset HEAD <file>
```

```sh
git mv old_filename new_filename #rename
git commit -m "Rename file"
git push
```

```sh
git add <files> #or git add .
git commit -m "initial commit"
git push
```

```sh
press "i" (i for insert)
write your merge message
press "esc" (escape)
write ":wq" (write & quit)
then press enter
```

```sh
git remote add origin <repo> #push an existing repository from the command line
git push -u origin master
git push ssh://<repo> master
```

```sh
#create a new repository on the command line
echo "# xxx" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin <repo>
git push -u origin main # master
```

```sh
git config --global http.proxy http://<some-domain>:8080
git config --global --get http.proxy
```


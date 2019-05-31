#!/bin/zsh
sed -e ':a;N;$!ba;/\[/s/\n/ /g' -e 's|,\[| |g' -e 's|\s\+|,|g' -e 's|]|\n|g' -i /shared/hogwild.logs/$1/conf*
sed -e 's|^,||' -e 's|,$||' -i /shared/hogwild.logs/$1/conf*

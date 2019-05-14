#!/bin/zsh
sed -e ':a;N;$!ba;/\[/s/\n/ /g' -e 's|,\[||g' -e 's|\s\+|,|g' -e 's|]|\n|g' -i /scratch/$1.hogwild/conf*
sed -e 's|^,||' -e 's|,$||' -i /scratch/$1.hogwild/conf*

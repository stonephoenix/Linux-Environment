getscp ()
{ echo $USER@$(hostname -i | awk '{print $1}'):$(readlink -f $1) ;}

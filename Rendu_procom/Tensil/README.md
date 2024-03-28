-Récupérer IP Pynq : 
	dmesg (pour récupérer le numéro du port USB)
  sudo screen /dev/tty/USB2 115200 (ou /dev/ttyUSB2)
→ Évidemment adapter le port USB
ifconfig (dans la nouvelle fenêtre, pour récupérer l’ip), pour quitter cette fenêtre Ctrl+Maj+A puis  quit 

-Lancer la Pynq : 
	ssh xilinx@10.29.227.75
password: xilinx
	→ Adapter l’IP de la Pynq avec ce qu’on a trouvé précédemment

-Lancer le docker : 
	sudo docker pull tensilai/tensil
sudo docker run -v $(pwd):/work -w /work -it tensilai/tensil bash

-Télécharger le fichier ONNX

-Placer le fichier sur le docker depuis le local :
sudo docker cp test.onnx 4ae1980efa10:/demo/models
	→Adapter le nom du fichier ainsi que l’identifiant du docker (attention parfois il ne place pas le fichier dans /demo/models)s

-Compiler avec tensil depuis le docker : 
	tensil compile -a /demo/arch/pynqz1.tarch -m /demo/models/LeeTens_10_test.onnx -o "Output" -s true -v true
	On a ajouté le fait qu’il soit verbose pour savoir où on en est, cela peut prendre un petit peu de temps

-Copier les fichiers obtenus après compilation (tprog, tdata, tmodel) en local : 
	sudo docker cp 4ae1980efa10:path_fichier_docker ~/path_local
	→Adapter le nom du fichier ainsi que l’identifiant du docker, répéter l’opération pour tous les fichiers (le .t* ne doit pas fonctionner dans ce cas là)

-Copier ces mêmes fichier sur la Pynq : 
	scp test_onnx_pynqz1.t* xilinx@10.29.227.75:
	→ Adapter l’IP de la Pynq et les noms des fichiers
	→ Répéter cette opération pour tous les fichiers nécessaires notamment les poids, les biais, le bitstream…

-Lancer le script python depuis xilinx !
	→ Si jamais le script est un Jupyter Notebook, on peut directement y accéder en allant sur http://10.29.227.75 (adapter l’IP), le mot de passe est le même que la xilinx

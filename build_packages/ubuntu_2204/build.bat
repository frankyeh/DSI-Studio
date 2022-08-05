docker build -o dsi_studio . --progress=plain
tar -a -c -f dsi_studio_ubuntu_22.04.zip dsi_studio 
rmdir /s /q dsi_studio

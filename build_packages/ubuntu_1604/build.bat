docker build -o dsi_studio . --progress=plain
pause
tar -a -c -f dsi_studio_ubuntu_16.04.zip dsi_studio 
rmdir /s /q dsi_studio

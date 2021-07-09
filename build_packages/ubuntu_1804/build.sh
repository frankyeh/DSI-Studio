#Use the new docker buildkit, which allows us to copy files out after completion
DOCKER_BUILDKIT=1 docker build -o dsi_studio .
zip -r dsi_studio_ubuntu_18.04.zip dsi_studio

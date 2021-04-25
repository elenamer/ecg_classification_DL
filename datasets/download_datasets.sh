root="../temp"
for db in 'mitdb'
do
wget -r -N -c -np --directory-prefix=$root "https://physionet.org/files/$db/1.0.0/"
mkdir "${root}/${db}"
mv "${root}/physionet.org/files/${db}/1.0.0"/* "${root}/${db}"
done
rm -r "${root}/physionet.org/"
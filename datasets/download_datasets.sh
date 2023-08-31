root="../data"


for db in 'mitdb' 'svdb' 'afdb' 'ltafdb' 'incartdb';
do
wget -r -N -c -np --directory-prefix=$root "https://physionet.org/files/$db/1.0.0/"
mkdir "${root}/${db}"
mv "${root}/physionet.org/files/${db}/1.0.0"/* "${root}/${db}"
done


wget -r -N -c -np --directory-prefix=$root "https://physionet.org/files/ptb-xl/1.0.1/"
mkdir "${root}/ptb-xl"
mv "${root}/physionet.org/files/ptb-xl/1.0.1"/* "${root}/ptb-xl"
rm -r "${root}/physionet.org/"


wget -r -N -c -np --directory-prefix="${root}/cpsc2018" "http://hhbucket.oss-cn-hongkong.aliyuncs.com/TrainingSet1.zip"
wget -r -N -c -np --directory-prefix="${root}/cpsc2018" "http://hhbucket.oss-cn-hongkong.aliyuncs.com/TrainingSet2.zip"
wget -r -N -c -np --directory-prefix="${root}/cpsc2018" "http://hhbucket.oss-cn-hongkong.aliyuncs.com/TrainingSet3.zip"
wget -r -N -c -np --directory-prefix="${root}/cpsc2018" "http://2018.icbeb.org/file/REFERENCE.csv"
mv "${root}/cpsc2018/hhbucket.oss-cn-hongkong.aliyuncs.com"/* "${root}/cpsc2018"
mv "${root}/cpsc2018/2018.icbeb.org/file"/* "${root}/cpsc2018"
rm -r "${root}/cpsc2018/hhbucket.oss-cn-hongkong.aliyuncs.com/"
rm -r "${root}/cpsc2018/2018.icbeb.org/"
unzip "${root}/cpsc2018/TrainingSet1.zip" -d "${root}/cpsc2018"
unzip "${root}/cpsc2018/TrainingSet2.zip" -d "${root}/cpsc2018"
unzip "${root}/cpsc2018/TrainingSet3.zip" -d "${root}/cpsc2018"
mkdir "${root}/cpsc2018/TrainingSet"
mv "${root}/cpsc2018/TrainingSet1"/* "${root}/cpsc2018/TrainingSet"
rm -r "${root}/cpsc2018/TrainingSet1"
mv "${root}/cpsc2018/TrainingSet2"/* "${root}/cpsc2018/TrainingSet"
rm -r "${root}/cpsc2018/TrainingSet2"
mv "${root}/cpsc2018/TrainingSet3"/* "${root}/cpsc2018/TrainingSet"
rm -r "${root}/cpsc2018/TrainingSet3"
rm "${root}/cpsc2018/TrainingSet1.zip"
rm "${root}/cpsc2018/TrainingSet2.zip"
rm "${root}/cpsc2018/TrainingSet3.zip"


wget -r -N -c -np --directory-prefix="${root}/irhythm-test" "https://irhythm.github.io/public_data/CARDIOL_MAY_2017.zip"
mv "${root}/irhythm-test/irhythm.github.io/public_data"/* "${root}/irhythm-test"
rm -r "${root}/irhythm-test/irhythm.github.io/"
unzip "${root}/irhythm-test/CARDIOL_MAY_2017.zip" -d "${root}/irhythm-test"
rm "${root}/irhythm-test/CARDIOL_MAY_2017.zip"


wget -r -N -c -np --directory-prefix="${root}/arr10000" "https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/15651326/ECGData.zip"
wget -r -N -c -np --directory-prefix="${root}/arr10000" "https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/15653771/Diagnostics.xlsx"
wget -r -N -c -np --directory-prefix="${root}/arr10000" "https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/15651296/RhythmNames.xlsx"
wget -r -N -c -np --directory-prefix="${root}/arr10000" "https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/15651293/ConditionNames.xlsx"
wget -r -N -c -np --directory-prefix="${root}/arr10000" "https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/15653762/AttributesDictionary.xlsx"
mv "${root}/arr10000/s3-eu-west-1.amazonaws.com/pfigshare-u-files/15651326/ECGData.zip" "${root}/arr10000"
mv "${root}/arr10000/s3-eu-west-1.amazonaws.com/pfigshare-u-files/15653771/Diagnostics.xlsx" "${root}/arr10000"
mv "${root}/arr10000/s3-eu-west-1.amazonaws.com/pfigshare-u-files/15651296/RhythmNames.xlsx" "${root}/arr10000"
mv "${root}/arr10000/s3-eu-west-1.amazonaws.com/pfigshare-u-files/15651293/ConditionNames.xlsx" "${root}/arr10000"
mv "${root}/arr10000/s3-eu-west-1.amazonaws.com/pfigshare-u-files/15653762/AttributesDictionary.xlsx" "${root}/arr10000"
unzip "${root}/arr10000/ECGData.zip" -d "${root}/arr10000"
rm -r "${root}/arr10000/s3-eu-west-1.amazonaws.com/"
rm "${root}/arr10000/ECGData.zip"


wget -r -N -c -np --directory-prefix="${root}/cinc2017" https://physionet.org/files/challenge-2017/1.0.0/training2017.zip?download
wget -r -N -c -np --directory-prefix="${root}/cinc2017" https://physionet.org/files/challenge-2017/1.0.0/REFERENCE-v3.csv?download
mv "${root}/cinc2017/physionet.org/files/challenge-2017/1.0.0"/* "${root}/cinc2017"
mv "${root}/cinc2017/training2017.zip?download" "${root}/cinc2017/training2017.zip" 
mv "${root}/cinc2017/REFERENCE-v3.csv?download" "${root}/cinc2017/REFERENCE-v3.csv" 

unzip "${root}/cinc2017/training2017.zip" -d "${root}/cinc2017"

rm -r "${root}/cinc2017/physionet.org/"
rm -r "${root}/cinc2017/training2017.zip"

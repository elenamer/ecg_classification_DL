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
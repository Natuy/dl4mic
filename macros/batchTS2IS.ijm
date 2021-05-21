arguments = getArgument();
args = split(arguments,",");

setOption("ExpandableArrays",true);

inputFolder = "";
if(args.length < 1){    inputFolder = getDirectory("Input Folder");
}else{  inputFolder = args[0];}

outputFolder = "";
if(args.length < 2){    outputFolder = getDirectory("Output Folder");
}else{  outputFolder = args[1];}

batchTiffStack2ImageSequence(inputFolder,outputFolder);

function batchTiffStack2ImageSequence(inputFolder,outputFolder){
	setBatchMode(true);
	filelist = getFileList(inputFolder);
	for (i = 0; i < lengthOf(filelist); i++) {
	    if (endsWith(filelist[i], ".tif")) { 
	        open(inputFolder + File.separator + filelist[i]);
	        if (nSlices>1) {
				run("Image Sequence... ", "format=TIFF save="+outputFolder);	
			} else {
				saveAs("tiff", outputFolder +File.separator+title);
			}
			close();
	    } 
	}
	setBatchMode(false);
}

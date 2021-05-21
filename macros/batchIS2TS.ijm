arguments = getArgument();
args = split(arguments,",");

setOption("ExpandableArrays",true);

inputFolder = "";
if(args.length < 1){    inputFolder = getDirectory("Input Folder");
}else{  inputFolder = args[0];}

outputFolder = "";
if(args.length < 2){    outputFolder = getDirectory("Output Folder");
}else{  outputFolder = args[1];}

if(args.length < 4){
    Dialog.create("Data Size");

    if(args.length<3){  Dialog.addNumber("Number of Rows",11);
    }else{  Dialog.addNumber("Number of Rows",parseInt(args[2]));}

    Dialog.addNumber("Number of Columns",11);

    Dialog.show();

    nbRow = Dialog.getNumber();
    nbCol = Dialog.getNumber();
}else{
    nbRow = parseInt(args[2]);
    nbCol = parseInt(args[3]);
}

batchImageSequence2TiffStack(inputFolder,outputFolder,nbRow,nbCol);

function batchImageSequence2TiffStack(inputFolder,outputFolder,nbRow,nbCol){
	setBatchMode(true);
	filelist = getFileList(inputFolder);
	for(r=0;r<nbRow;r++){
	    for(c=0;c<nbCol;c++){
	        Cell_Code = getTrapName(r,c);
            run("Image Sequence...", "open="+inputFolder+filelist[0]+" file=("+Cell_Code+"[0-9][0-9][0-9][0-9].tif) sort");
            saveAs("tiff", outputFolder +File.separator+Cell_Code);
	    }
	}

    setBatchMode(false);
}

function getTrapName(row,column){
	_ROW_NAMES = newArray("A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z");
	displayedcolumn= column+1;

	if(row<26){ name = ""+_ROW_NAMES[row]+"-"+displayedcolumn ;
	}else{  name = ""+_ROW_NAMES[row%26]+floor(row/26)+"-"+ displayedcolumn;}
	return name;
}


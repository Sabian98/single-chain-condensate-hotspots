// Render selected pages of a PDF to PNG with macOS PDFKit (no poppler on this Mac).
// osascript -l JavaScript tools/pdfpages.js in.pdf outdir 3,7
ObjC.import('PDFKit'); ObjC.import('AppKit');
function run(argv){
  var doc=$.PDFDocument.alloc.initWithURL($.NSURL.fileURLWithPath(argv[0])); var n=doc.pageCount;
  var want=argv[2].split(',').map(Number);
  for(var k=0;k<want.length;k++){var i=want[k]-1; if(i>=n) continue; var pg=doc.pageAtIndex(i); var b=pg.boundsForBox(0);
    var img=pg.thumbnailOfSizeForBox($.NSMakeSize(b.size.width*1.4,b.size.height*1.4),0);
    var rep=$.NSBitmapImageRep.imageRepWithData(img.TIFFRepresentation);
    rep.representationUsingTypeProperties(4,$()).writeToFileAtomically(argv[1]+'/p'+want[k]+'.png',true);}
  return 'pages '+n;
}

// Render page 1 of a PDF to PNG at a given dpi with macOS PDFKit (no poppler on this Mac).
// osascript -l JavaScript tools/pdf2png.js in.pdf out.png 300
ObjC.import('PDFKit'); ObjC.import('AppKit');
function run(argv){
  var doc=$.PDFDocument.alloc.initWithURL($.NSURL.fileURLWithPath(argv[0]));
  var pg=doc.pageAtIndex(0), b=pg.boundsForBox(1), k=Number(argv[2]||300)/72;   // 1 = crop box
  var img=pg.thumbnailOfSizeForBox($.NSMakeSize(Math.round(b.size.width*k),Math.round(b.size.height*k)),1);
  var rep=$.NSBitmapImageRep.imageRepWithData(img.TIFFRepresentation);
  rep.representationUsingTypeProperties(4,$()).writeToFileAtomically(argv[1],true);
  return argv[1];
}

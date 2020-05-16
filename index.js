#!/usr/bin/env node

const fs = require('fs')
if (process.argv.length<3){
  let help=`\
reconvert-vscode-py-ipynb <file name>
    The <file name> file should be a '.ipybn'
    file converted to '.py' for use in debugger.
    That file should have cells deliniated 
    by lines matching the regexp 
      "^# In\[.*\]:$"
    The text is converted back into '.ipynb' format
    (without text, of course), and written to 
    standard output.   
    This may be useful for comparing and merging changes.       
`
  console.log(help);
  process.exit(0);
}
const fnin=process.argv[2]
const out=process.stdout;
re=RegExp(/^# In\[.*\]:$/m)
var text=fs.readFileSync(fnin,'utf8')
var chunks=text.split(re)

var obj={
  cells:[],
  "metadata": {
    "kernelspec": {
     "display_name": "Python 3",
     "language": "python",
     "name": "python3"
    },
    "language_info": {
     "codemirror_mode": {
      "name": "ipython",
      "version": 3
     },
     "file_extension": ".py",
     "mimetype": "text/x-python",
     "name": "python",
     "nbconvert_exporter": "python",
     "pygments_lexer": "ipython3",
     "version": "3.7.6"
    }
   },
   "nbformat": 4,
   "nbformat_minor": 4  
}
for (const c of chunks){
  let cell = {
    cell_type : "code",
    source : [],
    execution_count: 1,
    metadata: {},
    outputs: [],  
  }
  let lines = c.split('\n');
  // remove upto three empty lines from beginning and end
  for (let i=0; i<3; i++) 
    if (i<lines.length && lines[0].length==0)
      lines.shift()
    else
      break
  for (let i=0; i<3; i++){
    let j = lines.length-1;
    if (j>=0 && lines[j].length==0)
      lines.splice(j,1)
    else
      break
  }
  for (let line of lines)
    cell.source.push(line+'\n'); 
  
  obj.cells.push(cell);
}

out.write(JSON.stringify(obj,null,2))



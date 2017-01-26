var td = require('./training-data');
var ann = require('./ann');
for (var key in td) module.exports[key] = td[key];
for (var key in ann) module.exports[key] = ann[key];


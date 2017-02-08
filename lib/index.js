// Copyright 2016 Zipscene, LLC
// Licensed under the Apache License, Version 2.0
// http://www.apache.org/licenses/LICENSE-2.0

var td = require('./training-data');
var ann = require('./ann');
for (var key in td) module.exports[key] = td[key];
for (var key in ann) module.exports[key] = ann[key];
module.exports.getAddon = require('./utils').getAddon;


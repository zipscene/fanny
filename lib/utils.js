// Copyright 2016 Zipscene, LLC
// Licensed under the Apache License, Version 2.0
// http://www.apache.org/licenses/LICENSE-2.0

var fs = require('fs');
var addonPathSet = [ '../build/Debug', '../build/Release' ];
var addonPath = addonPathSet[0];
for (var i = 0; i < addonPathSet.length; ++i) {
	if (fs.existsSync(addonPathSet[i])) {
		addonPath = addonPathSet[i];
		break;
	}
}

addonPath = addonPath + '/';

require(addonPath + 'addon-floatfann');

var XError = require('xerror');

function getAddon(datatype) {
	if (!datatype) datatype = 'float';
	if (datatype === 'float' || datatype === 'double' || datatype === 'fixed') {
		return require(addonPath + 'addon-' + datatype + 'fann');
	} else {
		throw new XError(XError.INVALID_ARGUMENT, 'Invalid FANN datatype: ' + datatype);
	}
}

module.exports = {
	getAddon: getAddon
};


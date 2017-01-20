{
	"targets": [
		{
			"target_name": "addon-floatfann",
			"sources": [
				"src/addon.cc",
				"src/fanny.cc",
				"src/utils.cc"
			],
			"libraries": [
				"-lfloatfann"
			],
			"include_dirs": [ "<!(node -e \"require('nan')\")" ],
			"defines": [
				"FANNY_FLOAT"
			]
		},
		{
			"target_name": "addon-doublefann",
			"sources": [
				"src/addon.cc",
				"src/fanny.cc",
				"src/utils.cc"
			],
			"libraries": [
				"-ldoublefann"
			],
			"include_dirs": [ "<!(node -e \"require('nan')\")" ],
			"defines": [
				"FANNY_DOUBLE"
			]
		},
		{
			"target_name": "addon-fixedfann",
			"sources": [
				"src/addon.cc",
				"src/fanny.cc",
				"src/utils.cc"
			],
			"libraries": [
				"-lfixedfann"
			],
			"include_dirs": [ "<!(node -e \"require('nan')\")" ],
			"defines": [
				"FANNY_FIXED"
			]
		}

	]
}

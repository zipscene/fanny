{
	"targets": [
		{
			"target_name": "addon",
			"sources": [
				"src/addon.cc",
				"src/fanny.cc"
			],
			"include_dirs": [ "<!(node -e \"require('nan')\")" ]
		}
	]
}

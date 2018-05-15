function fadein (object,mls) {
	if (object.style.opacity) {
		if (object.style.opacity>=1) return -1;
	//	var initialOpacity = 100;
	}
	var i = 0;
	var targetOpacity = 100;
	object.style.visibility = "visible";
	if (object.style.opacity==""||object.style.opacity==undefined) object.style.opacity=0;
	if (object.style.MozOpacity==""||object.style.MozOpacity==undefined) object.style.MozOpacity=0;
	if (object.style.filter=""||object.style.filter==undefined) object.style.filter = "progid:DXImageTransform.Microsoft.Alpha(opacity=0)";
	var intervalID = setInterval(function() {
		object.style.opacity = object.style.opacity * 1 + (targetOpacity/1000);
		object.style.MozOpacity = object.style.MozOpacity * 1 + (targetOpacity/1000);
		i = i + (targetOpacity/10);
		var buff = 'progid:DXImageTransform.Microsoft.Alpha(opacity=';
		buff += i;
		buff += ')';
		object.style.filter = buff;
		if (i == targetOpacity) {
			clearInterval(intervalID);
		}
	}, mls / 10);
};

function fadeout (object,mls) {
    if (object.style.opacity) {
        if (object.style.opacity<=0) return -1;
        var initialOpacity = 100;
    }
    else var initialOpacity = 100;
    var i = initialOpacity;
    if (object.style.opacity==""||object.style.opacity==undefined) object.style.opacity=initialOpacity/100;
    if (object.style.MozOpacity==""||object.style.MozOpacity==undefined) object.style.MozOpacity=initialOpacity/100;
    if (object.style.filter=""||object.style.filter==undefined) object.style.filter = "progid:DXImageTransform.Microsoft.Alpha(opacity="+initialOpacity+')';
    var intervalID = setInterval(function() {
	object.style.opacity = object.style.opacity * 1 - (initialOpacity/1000);
	object.style.MozOpacity = object.style.MozOpacity * 1 - (initialOpacity/1000);
	i = i - (initialOpacity/10);
	var buff = 'progid:DXImageTransform.Microsoft.Alpha(opacity=';
	buff += i;
	buff += ')';
	object.style.filter = buff;
	if (i == 0) {
	    clearInterval(intervalID);
	}
    }, mls / 10);
    setTimeout(function() {
        object.style.visibility = "hidden";
    }, mls);
}

/*
function hideobj(objid) {
	$(document).mouseup(function(e) 
	{
	    var container = $(objid);

	    // if the target of the click isn't the container nor a descendant of the container
	    if (!container.is(e.target) && container.has(e.target).length === 0) 
	    {
		container.hide();
		$(objid).unbind( 'click', clickDocument );
	    }
	});
};
*/

function showcut() {
//	document.documentElement.style.overflow="visible";
//	document.body.style.overflow="visible";
	document.getElementById("cut").style.display = "block";
	fadein(document.getElementById("cut"),500);
	fadeout(document.getElementById("showcut"),500);
	setTimeout(function() {
		document.getElementById("showcut").style.display = "none";
		document.getElementById("hidecut").style.display = "block";
		fadein(document.getElementById("hidecut"),500);
		rsz();
	}, 500);
}

function hidecut() {
//	document.documentElement.style.overflow="hidden";
//	document.body.style.overflow="hidden";
	fadeout(document.getElementById("cut"),500);
	fadeout(document.getElementById("hidecut"),500);
	setTimeout(function() {
		document.getElementById("hidecut").style.display = "none";
		document.getElementById("showcut").style.display = "block";
		fadein(document.getElementById("showcut"),500);
		document.getElementById("cut").style.display = "none";
		document.documentElement.style.overflow="auto";
		document.body.style.overflow="auto";
		rsz();
	}, 500);
}

function popuplength(event) {
	document.getElementById("length").style.display = "block";
	document.getElementById("length").style.left = event.pageX+"px";
/*
	if (document.getElementById("lengthlink").getBoundingClientRect().top+pageYOffset+document.getElementById("lengthlink").getBoundingClientRect().height/2<event.pageY) .............................
	else
	document.getElementById("length").style.top = document.getElementById("lengthlink").getBoundingClientRect().top+pageYOffset-1-document.getElementById("length").getBoundingClientRect().height + "px";
*/
	document.getElementById("length").style.top = document.getElementById("lengthlink").getBoundingClientRect().bottom+pageYOffset+2+"px";
	fadein(document.getElementById("length"),500);
	setTimeout(function() {
		document.getElementById("lengthlink").setAttribute("onclick","hidelength()");
	}, 500);
}

function hidelength() {
	fadeout(document.getElementById("length"),500);
	setTimeout(function() {
		document.getElementById("length").style.display = "none";
		document.getElementById("lengthlink").setAttribute("onclick","popuplength(event)");
	}, 500);
}

/*
function void0() {
//	document.getElementById("lengthlink").onclick=hidelength();
//	document.getElementById("lengthlink").href="javascript:void1()";
}

function void1() {
//	document.getElementById("lengthlink").onclick=popuplength();
//	document.getElementById("lengthlink").href="javascript:void0()";
}
*/

window.onload = function() {
//	document.getElementsByName('text_to_inspect')[0].value = decodeURIComponent(window.location.hash.replace('#', ''));

	var textarea = document.querySelector('textarea');

//	textarea.addEventListener('keydown', autosize);
		     
	function autosize() {
	  var el = this;
	  setTimeout(function() {
	    el.style.cssText = 'height:auto; padding:0';
	    // for box-sizing other than "content-box" use:
	    // el.style.cssText = '-moz-box-sizing:content-box';
	    el.style.cssText = 'height:' + el.scrollHeight + 'px';
	  },0);
	}
	
	rsz();

	fadein(document.body,500);
	
	/*
	var Red = "#CC0000";
	var Green = "#00CC00";
	document.getElementById("spelling_tr").style.color=Red;
	document.getElementById("academic_tr").style.color=Red;
	document.getElementById("verbs_tr").style.color=Red;
	document.getElementById("linking_tr").style.color=Green;
	document.getElementById("collocations_tr").style.color=Green;
	document.getElementById("length_tr").style.color=Red;
	*/
};

function showmasker(maskid) {
	var placerShown = true;
	var elements = document.getElementsByName('textmask');
	elements.forEach(function(item){
		if (item.style.visibility=="visible") {
			fadeout(item,500);
			placerShown = false;
			document.getElementById(item.id+'link').href="javascript:showmasker('"+item.id+"')";
		}
	});
	fadein(document.getElementById(maskid),500);
	if (placerShown) fadeout(document.getElementById("textplacer"),500);
	// hideobj(maskid);
}

function hidemasker(maskid) {
	fadeout(document.getElementById(maskid),500);
	fadein(document.getElementById("textplacer"),500);
	// hideobj(maskid);
}

function reversemaskerlink(caller,maskid) {
	if (document.getElementById(maskid).style.opacity==1) caller.href="javascript:hidemasker('"+maskid+"')";
	else caller.href="javascript:showmasker('"+maskid+"')";
}

/*
window.onresize = function() {
	var elements = document.getElementsByName('textmask');
	elements.forEach(function(item){
	item.style.width = document.getElementById("textplacer").clientWidth+'px';
	item.style.top = document.getElementById("textplacer").getBoundingClientRect().top+'px';
	item.style.left = document.getElementById("textplacer").getBoundingClientRect().left+'px';
	});
};
*/

function rsz() {
//	document.getElementById("textplacer").style.width = '98%';
	var elements = document.getElementsByName('textmask');
	elements.forEach(function(item){
	item.style.width = "100%";
//	item.style.top = Math.round(document.getElementById("textplacer").getBoundingClientRect().top)+'px';
//	item.style.left = Math.round(document.getElementById("textplacer").getBoundingClientRect().left)+'px';
	});
//	document.getElementById("textplacer").style.width = Math.round(document.getElementById("textplacer").clientWidth)+'px';
	if (document.getElementById("length").style.display!="none") hidelength();
	setTimeout(function() {
		var allelems = document.getElementsByTagName('*');
		Array.prototype.forEach.call(allelems, function(item){
		if (item.style.opacity)
			if (item.style.opacity<0)
				item.style.opacity=0;
		});
	},1000);
};

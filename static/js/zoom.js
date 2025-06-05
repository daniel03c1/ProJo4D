function changeImage(newSrc) {
	// Get the image and result elements
	const img = document.getElementById("myimage");
	const result = document.getElementById("myresult");
  
	// Change the image source
	img.src = newSrc;
  
	// Update the zoom functionality
	result.style.backgroundImage = `url('${newSrc}')`;
  
	// Reinitialize the zoom functionality
	imageZoom("myimage", "myresult");
  }
  
  function imageZoom(imgID, resultID) {
	const img = document.getElementById(imgID);
	const result = document.getElementById(resultID);
	const lens = document.createElement("div");
	lens.setAttribute("class", "img-zoom-lens");
  
	// Remove existing lens if it exists
	const existingLens = img.parentElement.querySelector(".img-zoom-lens");
	if (existingLens) {
	  existingLens.remove();
	}
  
	// Append the lens to the image-zoom-container
	img.parentElement.insertBefore(lens, img);
  
	// Get the background-size ratio
	const cx = result.offsetWidth / lens.offsetWidth;
	const cy = result.offsetHeight / lens.offsetHeight;
  
	result.style.backgroundImage = `url('${img.src}')`;
	result.style.backgroundSize = `${img.width * cx}px ${img.height * cy}px`;
  
	// Execute a function when the mouse moves over the image or the lens
	lens.addEventListener("mousemove", moveLens);
	img.addEventListener("mousemove", moveLens);
  
	// Touch functionality for mobile devices
	lens.addEventListener("touchmove", moveLens);
	img.addEventListener("touchmove", moveLens);
  
	function moveLens(e) {
	  e.preventDefault();
	  const pos = getCursorPos(e);
	  let x = pos.x - lens.offsetWidth / 2;
	  let y = pos.y - lens.offsetHeight / 2;
  
	  // Prevent the lens from being positioned outside the image
	  if (x > img.width - lens.offsetWidth) x = img.width - lens.offsetWidth;
	  if (x < 0) x = 0;
	  if (y > img.height - lens.offsetHeight) y = img.height - lens.offsetHeight;
	  if (y < 0) y = 0;
  
	  // Set the position of the lens
	  lens.style.left = `${x}px`;
	  lens.style.top = `${y}px`;
  
	  // Display what the lens "sees"
	  result.style.backgroundPosition = `-${x * cx}px -${y * cy}px`;
	}
  
	function getCursorPos(e) {
	  const rect = img.getBoundingClientRect();
	  return {
		x: e.pageX - rect.left - window.pageXOffset,
		y: e.pageY - rect.top - window.pageYOffset,
	  };
	}
  }
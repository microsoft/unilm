<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>3D Car Racing</title>

<style>
* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

body {
  overflow: hidden;
  background: #111;
  font-family: Arial, sans-serif;
  touch-action: none;
}

#game {
  position: fixed;
  inset: 0;
}

#hud {
  position: fixed;
  top: 15px;
  left: 15px;
  z-index: 10;
  color: white;
  font-size: 18px;
  line-height: 1.7;
  text-shadow: 2px 2px 3px #000;
}

#message {
  position: fixed;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-direction: column;
  z-index: 20;
  color: white;
  background: rgba(0,0,0,.65);
  text-align: center;
}

#message h1 {
  font-size: 42px;
  margin-bottom: 15px;
}

button {
  border: 0;
  border-radius: 15px;
  padding: 15px 25px;
  font-size: 20px;
  font-weight: bold;
  margin: 8px;
  background: white;
  color: #111;
}

#controls {
  position: fixed;
  bottom: 20px;
  left: 0;
  right: 0;
  z-index: 15;
  display: flex;
  justify-content: space-between;
  padding: 0 25px;
}

.controlGroup {
  display: flex;
  gap: 12px;
}

.control {
  width: 75px;
  height: 65px;
  border-radius: 18px;
  border: 2px solid white;
  background: rgba(0,0,0,.55);
  color: white;
  font-size: 28px;
}
</style>
</head>

<body>

<div id="game"></div>

<div id="hud">
  🏁 Level: <span id="level">1</span><br>
  ⏱️ Time: <span id="time">30</span><br>
  ⭐ Score: <span id="score">0</span><br>
  🚗 Speed: <span id="speed">0</span>
</div>

<div id="message">
  <h1>🏎️ 3D CAR RACING</h1>
  <p>Reach the finish before time runs out!</p>
  <button id="startBtn">START RACE</button>
</div>

<div id="controls">
  <div class="controlGroup">
    <button class="control" id="left">◀</button>
    <button class="control" id="right">▶</button>
  </div>

  <div class="controlGroup">
    <button class="control" id="brake">▼</button>
    <button class="control" id="boost">⚡</button>
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js"></script>

<script>
/* =========================
   BASIC THREE.JS SETUP
========================= */

const scene = new THREE.Scene();

scene.background = new THREE.Color(0x87ceeb);

scene.fog = new THREE.Fog(0x87ceeb, 40, 180);

const camera = new THREE.PerspectiveCamera(
  70,
  innerWidth / innerHeight,
  0.1,
  500
);

camera.position.set(0, 6, 12);

const renderer = new THREE.WebGLRenderer({
  antialias: true
});

renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));

document.getElementById("game").appendChild(renderer.domElement);


/* =========================
   LIGHT
========================= */

const ambient = new THREE.HemisphereLight(
  0xffffff,
  0x555555,
  2
);

scene.add(ambient);

const sun = new THREE.DirectionalLight(
  0xffffff,
  2
);

sun.position.set(20, 40, 20);
scene.add(sun);


/* =========================
   ROAD
========================= */

const roadWidth = 16;
const roadLength = 500;

const grass = new THREE.Mesh(
  new THREE.PlaneGeometry(300, 600),
  new THREE.MeshLambertMaterial({
    color: 0x238b45
  })
);

grass.rotation.x = -Math.PI / 2;
grass.position.y = -0.1;
grass.position.z = -180;

scene.add(grass);

const road = new THREE.Mesh(
  new THREE.PlaneGeometry(roadWidth, roadLength),
  new THREE.MeshLambertMaterial({
    color: 0x333333
  })
);

road.rotation.x = -Math.PI / 2;
road.position.z = -180;

scene.add(road);


/* =========================
   ROAD LINES
========================= */

const roadLines = [];

for (let z = 10; z > -500; z -= 12) {

  const line = new THREE.Mesh(
    new THREE.BoxGeometry(0.35, 0.05, 5),
    new THREE.MeshBasicMaterial({
      color: 0xffffff
    })
  );

  line.position.set(0, 0.03, z);

  scene.add(line);
  roadLines.push(line);
}


/* =========================
   ROAD SIDES
========================= */

for (let side of [-1, 1]) {

  const edge = new THREE.Mesh(
    new THREE.BoxGeometry(0.35, 0.08, roadLength),
    new THREE.MeshBasicMaterial({
      color: 0xff3333
    })
  );

  edge.position.set(
    side * (roadWidth / 2 - 0.2),
    0.04,
    -180
  );

  scene.add(edge);
}


/* =========================
   CAR CREATOR
========================= */

function createCar(color) {

  const car = new THREE.Group();

  // Body
  const body = new THREE.Mesh(
    new THREE.BoxGeometry(2.2, 0.7, 4),
    new THREE.MeshLambertMaterial({
      color: color
    })
  );

  body.position.y = 0.7;
  car.add(body);

  // Top
  const top = new THREE.Mesh(
    new THREE.BoxGeometry(1.55, 0.65, 1.8),
    new THREE.MeshLambertMaterial({
      color: color
    })
  );

  top.position.set(0, 1.25, -0.25);
  car.add(top);

  // Windows
  const windowMaterial = new THREE.MeshBasicMaterial({
    color: 0x111827
  });

  const frontWindow = new THREE.Mesh(
    new THREE.BoxGeometry(1.3, 0.35, 0.05),
    windowMaterial
  );

  frontWindow.position.set(0, 1.38, -1.18);
  car.add(frontWindow);

  const backWindow = new THREE.Mesh(
    new THREE.BoxGeometry(1.3, 0.35, 0.05),
    windowMaterial
  );

  backWindow.position.set(0, 1.38, 0.65);
  car.add(backWindow);

  // Wheels
  for (let x of [-1.05, 1.05]) {
    for (let z of [-1.25, 1.25]) {

      const wheel = new THREE.Mesh(
        new THREE.CylinderGeometry(
          0.42,
          0.42,
          0.35,
          16
        ),
        new THREE.MeshLambertMaterial({
          color: 0x111111
        })
      );

      wheel.rotation.z = Math.PI / 2;

      wheel.position.set(x, 0.45, z);

      car.add(wheel);
    }
  }

  return car;
}


/* =========================
   PLAYER
========================= */

const player = createCar(0x1565ff);

player.position.set(0, 0, 5);

scene.add(player);


/* =========================
   OPPONENTS
========================= */

const opponents = [];

function createOpponents() {

  opponents.forEach(car => scene.remove(car));
  opponents.length = 0;

  const colors = [
    0xff2222,
    0xffff00,
    0xff6600,
    0xaa22ff,
    0x00cc66
  ];

  for (let i = 0; i < 7; i++) {

    const car = createCar(
      colors[i % colors.length]
    );

    car.position.x =
      [-5, -2.5, 2.5, 5][
        Math.floor(Math.random() * 4)
      ];

    car.position.z = -30 - i * 45;

    car.userData.speed =
      0.06 + Math.random() * 0.08;

    scene.add(car);
    opponents.push(car);
  }
}


/* =========================
   GAME VARIABLES
========================= */

let level = 1;
let score = 0;
let timeLeft = 30;

let gameRunning = false;

let speed = 0;
let maxSpeed = 0.55;

let moveLeft = false;
let moveRight = false;
let boosting = false;
let braking = false;

let finishDistance = 500;


/* =========================
   START GAME
========================= */

function startGame() {

  level = 1;
  score = 0;
  timeLeft = 30;
  maxSpeed = 0.55;

  player.position.x = 0;
  player.position.z = 5;

  createOpponents();

  gameRunning = true;

  document.getElementById("message").style.display = "none";
}

document.getElementById("startBtn")
.addEventListener("click", startGame);


/* =========================
   CONTROLS
========================= */

function pressLeft(value) {
  moveLeft = value;
}

function pressRight(value) {
  moveRight = value;
}

document.getElementById("left")
.addEventListener("pointerdown", () => pressLeft(true));
document.getElementById("left")
.addEventListener("pointerup", () => pressLeft(false));
document.getElementById("left")
.addEventListener("pointercancel", () => pressLeft(false));

document.getElementById("right")
.addEventListener("pointerdown", () => pressRight(true));
document.getElementById("right")
.addEventListener("pointerup", () => pressRight(false));
document.getElementById("right")
.addEventListener("pointercancel", () => pressRight(false));

document.getElementById("boost")
.addEventListener("pointerdown", () => boosting = true);
document.getElementById("boost")
.addEventListener("pointerup", () => boosting = false);
document.getElementById("boost")
.addEventListener("pointercancel", () => boosting = false);

document.getElementById("brake")
.addEventListener("pointerdown", () => braking = true);
document.getElementById("brake")
.addEventListener("pointerup", () => braking = false);
document.getElementById("brake")
.addEventListener("pointercancel", () => braking = false);


/* Keyboard */

document.addEventListener("keydown", e => {

  if (e.key === "ArrowLeft" || e.key === "a")
    moveLeft = true;

  if (e.key === "ArrowRight" || e.key === "d")
    moveRight = true;

  if (e.key === "ArrowUp" || e.key === "w")
    boosting = true;

  if (e.key === "ArrowDown" || e.key === "s")
    braking = true;
});

document.addEventListener("keyup", e => {

  if (e.key === "ArrowLeft" || e.key === "a")
    moveLeft = false;

  if (e.key === "ArrowRight" || e.key === "d")
    moveRight = false;

  if (e.key === "ArrowUp" || e.key === "w")
    boosting = false;

  if (e.key === "ArrowDown" || e.key === "s")
    braking = false;
});


/* =========================
   COLLISION
========================= */

function collision(a, b) {

  const dx = Math.abs(a.position.x - b.position.x);
  const dz = Math.abs(a.position.z - b.position.z);

  return dx < 1.8 && dz < 3;
}


/* =========================
   LEVEL UP
========================= */

function nextLevel() {

  if (level >= 5) {

    gameRunning = false;

    showMessage(
      "🏆 YOU WIN!",
      "You completed all 5 levels!",
      "PLAY AGAIN"
    );

    return;
  }

  level++;

  timeLeft = Math.max(15, 30 - level * 3);

  maxSpeed += 0.08;

  score += 100;

  createOpponents();

  player.position.z = 5;

  showTemporaryLevel();
}


function showTemporaryLevel() {

  const oldText =
    document.getElementById("level").innerText;

  document.getElementById("level").innerText = level;

  setTimeout(() => {
    document.getElementById("level").innerText = level;
  }, 500);
}


/* =========================
   GAME OVER
========================= */

function gameOver(text) {

  gameRunning = false;

  showMessage(
    "💥 GAME OVER",
    text,
    "TRY AGAIN"
  );
}


function showMessage(title, subtitle, button) {

  const box = document.getElementById("message");

  box.style.display = "flex";

  box.innerHTML = `
    <h1>${title}</h1>
    <p>${subtitle}</p>
    <button id="restartBtn">${button}</button>
  `;

  document
    .getElementById("restartBtn")
    .onclick = startGame;
}


/* =========================
   TIMER
========================= */

let lastTime = performance.now();

function updateTimer(delta) {

  timeLeft -= delta / 1000;

  if (timeLeft <= 0) {

    timeLeft = 0;

    gameOver("⏱️ Time is over!");
  }

  document.getElementById("time").innerText =
    Math.ceil(timeLeft);
}


/* =========================
   GAME UPDATE
========================= */

function update(delta) {

  if (!gameRunning) return;

  updateTimer(delta);

  /* Steering */

  if (moveLeft)
    player.position.x -= 0.18;

  if (moveRight)
    player.position.x += 0.18;

  /* Road boundaries */

  player.position.x = THREE.MathUtils.clamp(
    player.position.x,
    -6.3,
    6.3
  );


  /* Speed */

  let targetSpeed = maxSpeed;

  if (boosting)
    targetSpeed *= 1.8;

  if (braking)
    targetSpeed *= 0.35;

  speed +=
    (targetSpeed - speed) * 0.08;


  /* Move road */

  roadLines.forEach(line => {

    line.position.z += speed * 1.7;

    if (line.position.z > 15)
      line.position.z -= 500;
  });


  /* Opponents */

  opponents.forEach(car => {

    car.position.z +=
      speed + car.userData.speed;

    if (car.position.z > 15) {

      car.position.z =
        -400 - Math.random() * 100;

      car.position.x =
        [-5, -2.5, 2.5, 5][
          Math.floor(Math.random() * 4)
        ];

      score += 10;
    }

    if (collision(player, car)) {

      gameOver("Your car crashed!");
    }
  });


  /* Progress */

  finishDistance -= speed;

  if (finishDistance <= 0) {

    finishDistance = 500;

    nextLevel();
  }


  document.getElementById("score").innerText =
    score;

  document.getElementById("speed").innerText =
    Math.round(speed * 180) + " km/h";


  /* Camera */

  camera.position.x +=
    (player.position.x * 0.45 - camera.position.x) * 0.08;

  camera.lookAt(
    player.position.x,
    0.8,
    player.position.z - 10
  );
}


/* =========================
   ANIMATION
========================= */

function animate(now) {

  requestAnimationFrame(animate);

  const delta = now - lastTime;
  lastTime = now;

  update(delta);

  renderer.render(scene, camera);
}

animate(performance.now());


/* =========================
   RESIZE
========================= */

window.addEventListener("resize", () => {

  camera.aspect =
    innerWidth / innerHeight;

  camera.updateProjectionMatrix();

  renderer.setSize(
    innerWidth,
    innerHeight
  );
});
</script>

</body>
</html>1.250.650.70.040.080.03

5.90910.167.9778.21.57.7532040346089415121.250.650.70.040.080.030.350.11.80.876,255

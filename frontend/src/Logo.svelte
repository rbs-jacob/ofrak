<style>
  .container {
    font-family: inherit;
    font-size: x-large;
    color: inherit;
    background: inherit;
    line-height: inherit;
    display: flex;
    flex-direction: row;
    flex-wrap: nowrap;
    align-items: center;
    justify-content: center;
    margin: 0 0 1rem 0;
  }

  .mask {
    width: 2em;
    height: 2em;
    margin: 5px;
    background: var(--animal-background);
    display: inline-block;
    mask-image: var(--animal-image-url);
    -webkit-mask-image: var(--animal-image-url);
    mask-size: contain;
    -webkit-mask-size: contain;
    mask-position: center;
    -webkit-mask-position: center;
    mask-repeat: no-repeat;
    -webkit-mask-repeat: no-repeat;
  }

  .shadow {
    filter: drop-shadow(1.5px 1.5px 1px var(--secondary-bg-color))
      drop-shadow(2px 2px 3px var(--secondary-bg-color));
  }
</style>

<script>
  import { animals } from "./animals.js";

  let currentAnimal = Math.floor(Math.random() * animals.length);
  let animal = animals[currentAnimal];
</script>

<div class="container">
  <div class="shadow">
    <div
      class="mask"
      style="--animal-background: {animal.color}; --animal-image-url: url('{animal.src}');"
    ></div>
  </div>
  <span>OFRAK</span>
</div>

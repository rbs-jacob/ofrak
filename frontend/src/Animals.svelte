<style>
  .container {
    position: absolute;
    bottom: 0;
    display: flex;
    flex-flow: row nowrap;
    justify-content: space-evenly;
    overflow: hidden;
  }

  .mask {
    width: 200px;
    height: 200px;
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
    object-fit: scale-down;
    transform: translate(0, 105%);
    transition: transform 0.5s;
  }

  .hover {
    transform: translate(0, 15%);
  }
</style>

<script>
  import { animals } from "./animals.js";

  export let x,
    selectedAnimal,
    visible = true;
  $: selectedAnimal = Math.floor((x / window.innerWidth) * 5);
</script>

{#if visible}
  <div class="container">
    {#each animals as animal, index}
      <div class="shadow" class:hover="{selectedAnimal === index}">
        <div
          class="mask"
          style="--animal-background: {animal.color}; --animal-image-url: url('{animal.src}');"
        ></div>
      </div>
    {/each}
  </div>
{/if}
